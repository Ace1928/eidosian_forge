import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def _cases_axes_tuple_length_mismatch():
    filter_func = ndimage.gaussian_filter
    kwargs = dict(radius=3, mode='constant', sigma=1.0, order=0)
    for key, val in kwargs.items():
        yield (filter_func, kwargs, key, val)
    filter_funcs = [ndimage.uniform_filter, ndimage.minimum_filter, ndimage.maximum_filter]
    kwargs = dict(size=3, mode='constant', origin=0)
    for filter_func in filter_funcs:
        for key, val in kwargs.items():
            yield (filter_func, kwargs, key, val)