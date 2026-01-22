import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestCsqrt:

    def test_simple(self):
        check_complex_value(np.sqrt, 1, 0, 1, 0)
        rres = 0.5 * np.sqrt(2)
        ires = rres
        check_complex_value(np.sqrt, 0, 1, rres, ires, False)
        check_complex_value(np.sqrt, -1, 0, 0, 1)

    def test_simple_conjugate(self):
        ref = np.conj(np.sqrt(complex(1, 1)))

        def f(z):
            return np.sqrt(np.conj(z))
        check_complex_value(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        check = check_complex_value
        f = np.sqrt
        check(f, np.PZERO, 0, 0, 0)
        check(f, np.NZERO, 0, 0, 0)
        check(f, 1, np.inf, np.inf, np.inf)
        check(f, -1, np.inf, np.inf, np.inf)
        check(f, np.PZERO, np.inf, np.inf, np.inf)
        check(f, np.NZERO, np.inf, np.inf, np.inf)
        check(f, np.inf, np.inf, np.inf, np.inf)
        check(f, -np.inf, np.inf, np.inf, np.inf)
        check(f, -np.nan, np.inf, np.inf, np.inf)
        check(f, 1, np.nan, np.nan, np.nan)
        check(f, -1, np.nan, np.nan, np.nan)
        check(f, 0, np.nan, np.nan, np.nan)
        check(f, -np.inf, 1, np.PZERO, np.inf)
        check(f, np.inf, 1, np.inf, np.PZERO)

        def _check_ninf_nan(dummy):
            msgform = 'csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)'
            z = np.sqrt(np.array(complex(-np.inf, np.nan)))
            with np.errstate(invalid='ignore'):
                if not (np.isnan(z.real) and np.isinf(z.imag)):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_nan(None)
        check(f, np.inf, np.nan, np.inf, np.nan)
        check(f, np.nan, 0, np.nan, np.nan)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, np.nan, np.nan, np.nan)