import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_stat_funcs_2d():
    a = np.array([[5, 6, 0, 0, 0], [8, 9, 0, 0, 0], [0, 0, 0, 3, 5]])
    lbl = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 2, 2]])
    mean = ndimage.mean(a, labels=lbl, index=[1, 2])
    assert_array_equal(mean, [7.0, 4.0])
    var = ndimage.variance(a, labels=lbl, index=[1, 2])
    assert_array_equal(var, [2.5, 1.0])
    std = ndimage.standard_deviation(a, labels=lbl, index=[1, 2])
    assert_array_almost_equal(std, np.sqrt([2.5, 1.0]))
    med = ndimage.median(a, labels=lbl, index=[1, 2])
    assert_array_equal(med, [7.0, 4.0])
    min = ndimage.minimum(a, labels=lbl, index=[1, 2])
    assert_array_equal(min, [5, 3])
    max = ndimage.maximum(a, labels=lbl, index=[1, 2])
    assert_array_equal(max, [9, 5])