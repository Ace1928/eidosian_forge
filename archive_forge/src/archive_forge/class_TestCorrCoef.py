import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestCorrCoef:
    A = np.array([[0.15391142, 0.18045767, 0.14197213], [0.70461506, 0.96474128, 0.27906989], [0.9297531, 0.32296769, 0.19267156]])
    B = np.array([[0.10377691, 0.5417086, 0.49807457], [0.82872117, 0.77801674, 0.39226705], [0.9314666, 0.66800209, 0.03538394]])
    res1 = np.array([[1.0, 0.9379533, -0.04931983], [0.9379533, 1.0, 0.30007991], [-0.04931983, 0.30007991, 1.0]])
    res2 = np.array([[1.0, 0.9379533, -0.04931983, 0.30151751, 0.66318558, 0.51532523], [0.9379533, 1.0, 0.30007991, -0.04781421, 0.88157256, 0.78052386], [-0.04931983, 0.30007991, 1.0, -0.96717111, 0.71483595, 0.83053601], [0.30151751, -0.04781421, -0.96717111, 1.0, -0.51366032, -0.66173113], [0.66318558, 0.88157256, 0.71483595, -0.51366032, 1.0, 0.98317823], [0.51532523, 0.78052386, 0.83053601, -0.66173113, 0.98317823, 1.0]])

    def test_non_array(self):
        assert_almost_equal(np.corrcoef([0, 1, 0], [1, 0, 1]), [[1.0, -1.0], [-1.0, 1.0]])

    def test_simple(self):
        tgt1 = corrcoef(self.A)
        assert_almost_equal(tgt1, self.res1)
        assert_(np.all(np.abs(tgt1) <= 1.0))
        tgt2 = corrcoef(self.A, self.B)
        assert_almost_equal(tgt2, self.res2)
        assert_(np.all(np.abs(tgt2) <= 1.0))

    def test_ddof(self):
        with suppress_warnings() as sup:
            warnings.simplefilter('always')
            assert_warns(DeprecationWarning, corrcoef, self.A, ddof=-1)
            sup.filter(DeprecationWarning)
            assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)
            assert_almost_equal(corrcoef(self.A, ddof=3), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=3), self.res2)

    def test_bias(self):
        with suppress_warnings() as sup:
            warnings.simplefilter('always')
            assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)
            assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)
            sup.filter(DeprecationWarning)
            assert_almost_equal(corrcoef(self.A, bias=1), self.res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)
        tgt = np.array([[1.0, -1j], [1j, 1.0]])
        assert_allclose(res, tgt)
        assert_(np.all(np.abs(res) <= 1.0))

    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(np.corrcoef(x, y), np.array([[1.0, -1j], [1j, 1.0]]))

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', RuntimeWarning)
            assert_array_equal(corrcoef(np.array([])), np.nan)
            assert_array_equal(corrcoef(np.array([]).reshape(0, 2)), np.array([]).reshape(0, 0))
            assert_array_equal(corrcoef(np.array([]).reshape(2, 0)), np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    def test_extreme(self):
        x = [[1e-100, 1e+100], [1e+100, 1e-100]]
        with np.errstate(all='raise'):
            c = corrcoef(x)
        assert_array_almost_equal(c, np.array([[1.0, -1.0], [-1.0, 1.0]]))
        assert_(np.all(np.abs(c) <= 1.0))

    @pytest.mark.parametrize('test_type', [np.half, np.single, np.double, np.longdouble])
    def test_corrcoef_dtype(self, test_type):
        cast_A = self.A.astype(test_type)
        res = corrcoef(cast_A, dtype=test_type)
        assert test_type == res.dtype