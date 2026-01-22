import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestCabs:

    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1 + 1j, 0 + 2j, 1 + 2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.0), 2, np.sqrt(5), np.inf, np.nan])
        y = np.abs(x)
        assert_almost_equal(y, y_r)

    def test_fabs(self):
        x = np.array([1 + 0j], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(1, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(np.inf, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))
        x = np.array([complex(np.nan, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

    def test_cabs_inf_nan(self):
        x, y = ([], [])
        x.append(np.nan)
        y.append(np.nan)
        check_real_value(np.abs, np.nan, np.nan, np.nan)
        x.append(np.nan)
        y.append(-np.nan)
        check_real_value(np.abs, -np.nan, np.nan, np.nan)
        x.append(np.inf)
        y.append(np.nan)
        check_real_value(np.abs, np.inf, np.nan, np.inf)
        x.append(-np.inf)
        y.append(np.nan)
        check_real_value(np.abs, -np.inf, np.nan, np.inf)

        def f(a):
            return np.abs(np.conj(a))

        def g(a, b):
            return np.abs(complex(a, b))
        xa = np.array(x, dtype=complex)
        assert len(xa) == len(x) == len(y)
        for xi, yi in zip(x, y):
            ref = g(xi, yi)
            check_real_value(f, xi, yi, ref)