import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def to_tree(Z, rd=False):
    """
    Convert a linkage matrix into an easy-to-use tree object.

    The reference to the root `ClusterNode` object is returned (by default).

    Each `ClusterNode` object has a ``left``, ``right``, ``dist``, ``id``,
    and ``count`` attribute. The left and right attributes point to
    ClusterNode objects that were combined to generate the cluster.
    If both are None then the `ClusterNode` object is a leaf node, its count
    must be 1, and its distance is meaningless but set to 0.

    *Note: This function is provided for the convenience of the library
    user. ClusterNodes are not used as input to any of the functions in this
    library.*

    Parameters
    ----------
    Z : ndarray
        The linkage matrix in proper form (see the `linkage`
        function documentation).
    rd : bool, optional
        When False (default), a reference to the root `ClusterNode` object is
        returned.  Otherwise, a tuple ``(r, d)`` is returned. ``r`` is a
        reference to the root node while ``d`` is a list of `ClusterNode`
        objects - one per original entry in the linkage matrix plus entries
        for all clustering steps. If a cluster id is
        less than the number of samples ``n`` in the data that the linkage
        matrix describes, then it corresponds to a singleton cluster (leaf
        node).
        See `linkage` for more information on the assignment of cluster ids
        to clusters.

    Returns
    -------
    tree : ClusterNode or tuple (ClusterNode, list of ClusterNode)
        If ``rd`` is False, a `ClusterNode`.
        If ``rd`` is True, a list of length ``2*n - 1``, with ``n`` the number
        of samples.  See the description of `rd` above for more details.

    See Also
    --------
    linkage, is_valid_linkage, ClusterNode

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> x = rng.random((5, 2))
    >>> Z = hierarchy.linkage(x)
    >>> hierarchy.to_tree(Z)
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
    >>> rootnode
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> len(nodelist)
    9

    """
    xp = array_namespace(Z)
    Z = as_xparray(Z, order='c', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    d = [None] * (n * 2 - 1)
    for i in range(0, n):
        d[i] = ClusterNode(i)
    nd = None
    for i in range(Z.shape[0]):
        row = Z[i, :]
        fi = int_floor(row[0], xp)
        fj = int_floor(row[1], xp)
        if fi > i + n:
            raise ValueError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 0' % fi)
        if fj > i + n:
            raise ValueError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 1' % fj)
        nd = ClusterNode(i + n, d[fi], d[fj], row[2])
        if row[3] != nd.count:
            raise ValueError('Corrupt matrix Z. The count Z[%d,3] is incorrect.' % i)
        d[n + i] = nd
    if rd:
        return (nd, d)
    else:
        return nd