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
def test_uniform_filter1d(self):
    d = numpy.random.randn(5000)
    os = numpy.empty((4, d.size))
    ot = numpy.empty_like(os)
    self.check_func_serial(4, ndimage.uniform_filter1d, (d, 5), os)
    self.check_func_thread(4, ndimage.uniform_filter1d, (d, 5), ot)
    assert_array_equal(os, ot)