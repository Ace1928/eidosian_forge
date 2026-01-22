import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.

    In the context of a histogram, this makes the last bin edge inclusive
    """
    return np.concatenate((a.searchsorted(v[:-1], 'left'), a.searchsorted(v[-1:], 'right')))