import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _mwu_choose_method(n1, n2, xy, method):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""
    if n1 > 8 and n2 > 8:
        return 'asymptotic'
    if np.apply_along_axis(_tie_check, -1, xy).any():
        return 'asymptotic'
    return 'exact'