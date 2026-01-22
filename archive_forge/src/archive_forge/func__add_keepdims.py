import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def _add_keepdims(func):
    """ hack in keepdims behavior into a function taking an axis """

    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0
        return np.expand_dims(res, axis=axis)
    return wrapped