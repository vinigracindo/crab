# -*-coding:utf8-*-
""" Utilities to evaluate pairwise distances or metrics between 2
sets of points.

"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

import numpy as np
import scipy.spatial.distance as ssd

# Utility Function
def check_pairwise_arrays(X, Y, dtype=np.float):
    """ Set X and Y appropriately and checks inputs
    
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.
    
    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional. 
    -Finally, the function checks that the size
    of the second dimension of the two arrays is equal.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples_1, n_features]

    Y : {array-like, sparse matrix}, shape = [n_samples_2, n_features]
    
    Returns
    -------
    X : {array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guarenteed to be a numpy array.
        
    Y : {array-like, sparse matrix}, shape = [n_samples_a, n_features]
        An array equal to X, guarenteed to be a numpy array.
    
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as Y, then you should just pre-compute the output and not even
    # call this function.
    if X is Y:
        X = Y = np.asanyarray(X, dtype=dtype)
    else:
        X = np.asanyarray(X, dtype=dtype)
        Y = np.asanyarray(Y, dtype=dtype)
        
        # If X is Y this check will always be false.
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Incompatible dimension for X and Y matrices.")
    
    return X, Y

def euclidean_distances(X, Y, squared=False, inverse=True):
    """ 
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    An implementation of a "similarity" based on the Euclidean "distance"
    between two vectors X and Y. Thinking of items as dimensions and
    preferences as points along those dimensions, a distance is computed using
    all items (dimensions) where both users have expressed a preference for
    that item. This is simply the square root of the sum of the squares of
    differences in position (preference) along each dimension.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    squared: boolean, optional
        This routine will return squared Euclidean distances instead.

    inverse: boolean, optional
        This routine will return the inverse Euclidean distances instead.

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise import euclidean_distances
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[ 1.        ,  0.29429806],
           [ 0.29429806,  1.        ]])
    >>> # get distance to origin
    >>> X = [[1.0, 0.0],[1.0,1.0]]
    >>> euclidean_distances(X, [[0.0, 0.0]])
    array([[ 0.5       ],
          [ 0.41421356]])

    """
    X, Y = check_pairwise_arrays(X , Y)
    
    if squared:
        XY = ssd.cdist(X, Y, 'sqeuclidean')
    else:
        XY = ssd.cdist(X, Y, 'euclidean')
        
    return np.divide(1.0, (1.0 + XY)) if inverse else XY

euclidian_distances = euclidean_distances  # both spelling for backward compat

def pearson_correlation(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise import pearson_correlation
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> pearson_correlation(X, X)
    array([[ 1., 1.],
           [ 1., 1.]])
    >>> pearson_correlation(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.39605902],
               [ 0.39605902]])
    """
    X, Y = check_pairwise_arrays(X, Y)

    XY = ssd.cdist(X, Y, 'correlation', 2)

    return 1 - XY

def jaccard_coefficient(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is a statistic used for comparing the
    similarity and diversity of sample sets.
    The Jaccard coefficient measures similarity between sample sets,
    and is defined as the size of the intersection divided by the size of the
    union of the sample sets.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from scikits.crab.metrics.pairwise import jaccard_coefficient
    >>> X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    >>> # distance between rows of X
    >>> jaccard_coefficient(X, X)
    array([[ 1.,  0.],
           [ 0.,  1.]])

    >>> jaccard_coefficient(X, [['a', 'b', 'c', 'k']])
    array([[ 0.6],
           [ 0. ]])
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as Y, then you should just pre-compute the output and not even
    # call this function.

    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    #TODO: Check if it is possible to optimize this function
    sX = X.shape[0]
    sY = Y.shape[0]
    dm = np.zeros((sX, sY))
    for i in xrange(0, sX):
        for j in xrange(0, sY):
            sx = set(X[i])
            sy = set(Y[j])
            n_XY = len(sx & sy)
            d_XY = len(sx | sy)
            dm[i,j] = n_XY / float(d_XY) 
    return dm

def manhattan_distances(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This distance implementation is the distance between two points in a grid
    based on a strictly horizontal and/or vertical path (that is, along the
    grid lines as opposed to the diagonal or "as the crow flies" distance.
    The Manhattan distance is the simple sum of the horizontal and vertical
    components, whereas the diagonal distance might be computed by applying the
    Pythagorean theorem.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from scikits.crab.metrics.pairwise  import manhattan_distances
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> manhattan_distances(X, X)
    array([[ 1.,  1.],
           [ 1.,  1.]])
    >>> manhattan_distances(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.25],
          [ 0.25]])
    """
    X, Y = check_pairwise_arrays(X, Y)
    
    XY = ssd.cdist(X, Y, 'cityblock')

    return 1.0 - (XY / float(X.shape[1]))

def sorensen_coefficient(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    The Sørensen index, also known as Sørensen’s similarity coefficient,
    is a statistic used for comparing the similarity of two samples.
    It was developed by the botanist Thorvald Sørensen and published in 1948.
    [1]
    See the link:http://en.wikipedia.org/wiki/S%C3%B8rensen_similarity_index

    This is intended for "binary" data sets where a user either expresses a
    generic "yes" preference for an item or has no preference. The actual
    preference values do not matter here, only their presence or absence.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise import sorensen_coefficient
    >>> X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    >>> # distance between rows of X
    >>> sorensen_coefficient(X, X)
    array([[ 1.,  0.],
          [ 0.,  1.]])
    >>> sorensen_coefficient(X, [['a', 'b', 'c', 'k']])
    array([[ 0.75], [ 0.  ]])

    """
    # should not need X_norm_squared because if you could precompute that as
    # well as Y, then you should just pre-compute the output and not even
    # call this function.
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    sX = X.shape[0]
    sY = Y.shape[0]
    dm = np.zeros((sX, sY))
    
    #TODO: Check if it is possible to optimize this function
    for i in xrange(0, sX):
        for j in xrange(0, sY):
            sx = set(X[i])
            sy = set(Y[j])
            n_XY = len(sx & sy)
            dm[i,j] = (2.0 * n_XY) / (len(X[i]) + len(Y[j]))
            
    return dm

def tanimoto_coefficient(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    An implementation of a "similarity" based on the Tanimoto coefficient,
    or extended Jaccard coefficient.

    This is intended for "binary" data sets where a user either expresses a
    generic "yes" preference for an item or has no preference. The actual
    preference values do not matter here, only their presence or absence.

    Parameters
    ----------
    X: array of shape n_samples_1

    Y: array of shape n_samples_2

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise  import tanimoto_coefficient
    >>> X =  [['a', 'b', 'c', 'd'],['e', 'f','g']]
    >>> # distance between rows of X
    >>> tanimoto_coefficient(X, X)
    array([[ 1.,  0.],
           [ 0.,  1.]])
    >>> tanimoto_coefficient(X, [['a', 'b', 'c', 'k']])
    array([[ 0.6],
           [ 0. ]])

    """
    return jaccard_coefficient(X,Y)

def cosine_distances(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

     An implementation of the cosine similarity. The result is the cosine of
     the angle formed between the two preference vectors.
     Note that this similarity does not "center" its data, shifts the user's
     preference values so that each of their means is 0. For this behavior,
     use Pearson Coefficient, which actually is mathematically
     equivalent for centered data.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise  import cosine_distances
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> cosine_distances(X, X)
    array([[ 1.,  1.],
          [ 1.,  1.]])
    >>> cosine_distances(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.9606463],
           [ 0.9606463]])

    """
    X, Y = check_pairwise_arrays(X, Y)

    return 1. - ssd.cdist(X, Y, 'cosine')

def spearman_coefficient(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    Like  Pearson Coefficient , but compares relative ranking of preference
    values instead of preference values themselves. That is, each user's
    preferences are sorted and then assign a rank as their preference value,
    with 1 being assigned to the least preferred item.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise  import spearman_coefficient
    >>> X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5)],[ ('e', 2.5),('f', 3.0), ('g', 2.5), ('h', 4.0)] ]
    >>> # distance between rows of X
    >>> spearman_coefficient(X, X)
    array([[ 1.,  0.],
           [ 0.,  1.]])
    >>> spearman_coefficient(X, [[('a',2.5),('b', 3.5), ('c',3.0), ('k',3.5)]])
    array([[ 1.],
           [ 0.]])
    """
    X, Y = check_pairwise_arrays(X, Y, dtype=[('x', 'S30'), ('y', np.float)])

    X.sort(order='y')
    Y.sort(order='y')

    result = []

    #TODO: Check if it is possible to optimize this function
    i = 0
    for arrayX in X:
        result.append([])
        for arrayY in Y:
            Y_keys = [key for key, value in arrayY]

            XY = [(key, value) for key, value in arrayX if key in Y_keys]
            
            n = len(XY)
            if n == 0:
                result[i].append(0.0)
            else:
                sumDiffSq = 0.0
                for index, tup in enumerate(XY):
                    sumDiffSq += pow((index + 1) - (Y_keys.index(tup[0]) + 1), 2.0)
                result[i].append(1.0 - ((6.0 * sumDiffSq) / (n * (n * n - 1))))
                
        result[i] = np.asanyarray(result[i])
        i += 1

    return np.asanyarray(result)


def loglikehood_coefficient(n_items, X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    Parameters
    ----------
    n_items: int
        Number of items in the model.

    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from crab.metrics.pairwise import loglikehood_coefficient
    >>> X = [['a', 'b', 'c', 'd'],  ['e', 'f','g', 'h']]
    >>> # distance between rows of X
    >>> n_items = 7
    >>> loglikehood_coefficient(n_items,X, X)
    array([[ 1.,  0.],
          [ 0.,  1.]])
    >>> n_items = 8
    >>> loglikehood_coefficient(n_items, X, [['a', 'b', 'c', 'k']])
    array([[ 0.67668852],
          [ 0.        ]])


    References
    ----------
    See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.5962 and
    http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html.
    """

    def safeLog(d):
        if d <= 0.0:
            return 0.0
        else:
            return np.log(d)

    def logL(p, k, n):
        return k * safeLog(p) + (n - k) * safeLog(1.0 - p)

    def twoLogLambda(k1, k2, n1, n2):
        p = (k1 + k2) / (n1 + n2)
        return 2.0 * (logL(k1 / n1, k1, n1) + logL(k2 / n2, k2, n2)
                      - logL(p, k1, n1) - logL(p, k2, n2))

    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    result = []

    # TODO: Check if it is possible to optimize this function

    i = 0
    for arrayX in X:
        result.append([])
        for arrayY in Y:
            XY = np.intersect1d(arrayX, arrayY)

            if XY.size == 0:
                result[i].append(0.0)
            else:
                nX = arrayX.size
                nY = arrayY.size
                if (nX - XY.size == 0)  or (n_items - nY) == 0:
                    result[i].append(1.0)
                else:
                    logLikelihood = twoLogLambda(float(XY.size),
                                                 float(nX - XY.size),
                                                 float(nY),
                                                 float(n_items - nY))

                    result[i].append(1.0 - 1.0 / (1.0 + float(logLikelihood)))
        result[i] = np.asanyarray(result[i])
        i += 1

    return np.asanyarray(result)