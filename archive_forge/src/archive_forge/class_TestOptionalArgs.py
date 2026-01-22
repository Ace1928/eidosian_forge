import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestOptionalArgs:

    def test_ndarrayfuncs(self):
        d = np.arange(24.0).reshape((2, 3, 4))
        m = np.zeros(24, dtype=bool).reshape((2, 3, 4))
        m[:, :, -1] = True
        a = np.ma.array(d, mask=m)

        def testaxis(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)
            assert_equal(ma_f(a, axis=1)[..., :-1], numpy_f(d[..., :-1], axis=1))
            assert_equal(ma_f(a, axis=(0, 1))[..., :-1], numpy_f(d[..., :-1], axis=(0, 1)))

        def testkeepdims(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)
            assert_equal(ma_f(a, keepdims=True).shape, numpy_f(d, keepdims=True).shape)
            assert_equal(ma_f(a, keepdims=False).shape, numpy_f(d, keepdims=False).shape)
            assert_equal(ma_f(a, axis=1, keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=1, keepdims=True))
            assert_equal(ma_f(a, axis=(0, 1), keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=(0, 1), keepdims=True))
        for f in ['sum', 'prod', 'mean', 'var', 'std']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)
        for f in ['min', 'max']:
            testaxis(f, a, d)
        d = np.arange(24).reshape((2, 3, 4)) % 2 == 0
        a = np.ma.array(d, mask=m)
        for f in ['all', 'any']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

    def test_count(self):
        d = np.arange(24.0).reshape((2, 3, 4))
        m = np.zeros(24, dtype=bool).reshape((2, 3, 4))
        m[:, 0, :] = True
        a = np.ma.array(d, mask=m)
        assert_equal(count(a), 16)
        assert_equal(count(a, axis=1), 2 * ones((2, 4)))
        assert_equal(count(a, axis=(0, 1)), 4 * ones((4,)))
        assert_equal(count(a, keepdims=True), 16 * ones((1, 1, 1)))
        assert_equal(count(a, axis=1, keepdims=True), 2 * ones((2, 1, 4)))
        assert_equal(count(a, axis=(0, 1), keepdims=True), 4 * ones((1, 1, 4)))
        assert_equal(count(a, axis=-2), 2 * ones((2, 4)))
        assert_raises(ValueError, count, a, axis=(1, 1))
        assert_raises(np.AxisError, count, a, axis=3)
        a = np.ma.array(d, mask=nomask)
        assert_equal(count(a), 24)
        assert_equal(count(a, axis=1), 3 * ones((2, 4)))
        assert_equal(count(a, axis=(0, 1)), 6 * ones((4,)))
        assert_equal(count(a, keepdims=True), 24 * ones((1, 1, 1)))
        assert_equal(np.ndim(count(a, keepdims=True)), 3)
        assert_equal(count(a, axis=1, keepdims=True), 3 * ones((2, 1, 4)))
        assert_equal(count(a, axis=(0, 1), keepdims=True), 6 * ones((1, 1, 4)))
        assert_equal(count(a, axis=-2), 3 * ones((2, 4)))
        assert_raises(ValueError, count, a, axis=(1, 1))
        assert_raises(np.AxisError, count, a, axis=3)
        assert_equal(count(np.ma.masked), 0)
        assert_raises(np.AxisError, count, np.ma.array(1), axis=1)