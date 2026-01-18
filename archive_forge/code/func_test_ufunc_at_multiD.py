import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_ufunc_at_multiD(self):
    a = np.arange(9).reshape(3, 3)
    b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
    np.add.at(a, (slice(None), [1, 2, 1]), b)
    assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])
    a = np.arange(27).reshape(3, 3, 3)
    b = np.array([100, 200, 300])
    np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
    assert_equal(a, [[[0, 401, 202], [3, 404, 205], [6, 407, 208]], [[9, 410, 211], [12, 413, 214], [15, 416, 217]], [[18, 419, 220], [21, 422, 223], [24, 425, 226]]])
    a = np.arange(9).reshape(3, 3)
    b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
    np.add.at(a, ([1, 2, 1], slice(None)), b)
    assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])
    a = np.arange(27).reshape(3, 3, 3)
    b = np.array([100, 200, 300])
    np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
    assert_equal(a, [[[0, 1, 2], [203, 404, 605], [106, 207, 308]], [[9, 10, 11], [212, 413, 614], [115, 216, 317]], [[18, 19, 20], [221, 422, 623], [124, 225, 326]]])
    a = np.arange(9).reshape(3, 3)
    b = np.array([100, 200, 300])
    np.add.at(a, (0, [1, 2, 1]), b)
    assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])
    a = np.arange(27).reshape(3, 3, 3)
    b = np.array([100, 200, 300])
    np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
    assert_equal(a, [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[209, 410, 611], [12, 13, 14], [15, 16, 17]], [[118, 219, 320], [21, 22, 23], [24, 25, 26]]])
    a = np.arange(27).reshape(3, 3, 3)
    b = np.array([100, 200, 300])
    np.add.at(a, (slice(None), slice(None), slice(None)), b)
    assert_equal(a, [[[100, 201, 302], [103, 204, 305], [106, 207, 308]], [[109, 210, 311], [112, 213, 314], [115, 216, 317]], [[118, 219, 320], [121, 222, 323], [124, 225, 326]]])