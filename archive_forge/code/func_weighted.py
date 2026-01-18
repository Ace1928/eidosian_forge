import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def weighted(y):
    """
    Perform weighted/WPGMA linkage on the condensed distance matrix.

    See `linkage` for more information on the return
    structure and algorithm.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        `linkage` for more information on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import weighted, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = weighted(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 11.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.20710678,  3.        ],
           [ 8.        , 13.        ,  1.20710678,  3.        ],
           [ 5.        , 14.        ,  1.20710678,  3.        ],
           [10.        , 15.        ,  1.20710678,  3.        ],
           [18.        , 19.        ,  3.05595762,  6.        ],
           [16.        , 17.        ,  3.32379407,  6.        ],
           [20.        , 21.        ,  4.06357713, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9,  1,  2,  3, 10, 11, 12,  4,  6,  5], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([3, 3, 3, 1, 1, 1, 4, 4, 4, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    array([2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    return linkage(y, method='weighted', metric='euclidean')