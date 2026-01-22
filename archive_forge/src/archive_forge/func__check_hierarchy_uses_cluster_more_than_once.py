import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _check_hierarchy_uses_cluster_more_than_once(Z):
    n = Z.shape[0] + 1
    chosen = set()
    for i in range(0, n - 1):
        used_more_than_once = float(Z[i, 0]) in chosen or float(Z[i, 1]) in chosen or Z[i, 0] == Z[i, 1]
        if used_more_than_once:
            return True
        chosen.add(float(Z[i, 0]))
        chosen.add(float(Z[i, 1]))
    return False