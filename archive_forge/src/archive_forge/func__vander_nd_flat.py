import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis

    Used to implement the public ``<type>vander<n>d`` functions.
    """
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[:-len(degrees)] + (-1,))