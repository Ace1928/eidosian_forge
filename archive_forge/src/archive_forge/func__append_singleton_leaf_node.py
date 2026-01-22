import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func, i, labels):
    if lvs is not None:
        lvs.append(int(i))
    if ivl is not None:
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        elif labels is not None:
            ivl.append(labels[int(i - n)])
        else:
            ivl.append(str(int(i)))