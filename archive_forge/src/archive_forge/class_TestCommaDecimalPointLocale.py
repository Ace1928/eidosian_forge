import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    def test_repr_roundtrip_foreign(self):
        o = 1.5
        assert_equal(o, np.longdouble(repr(o)))

    def test_fromstring_foreign_repr(self):
        f = 1.234
        a = np.fromstring(repr(f), dtype=float, sep=' ')
        assert_equal(a[0], f)

    def test_fromstring_best_effort_float(self):
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1,234', dtype=float, sep=' '), np.array([1.0]))

    def test_fromstring_best_effort(self):
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1,234', dtype=np.longdouble, sep=' '), np.array([1.0]))

    def test_fromstring_foreign(self):
        s = '1.234'
        a = np.fromstring(s, dtype=np.longdouble, sep=' ')
        assert_equal(a[0], np.longdouble(s))

    def test_fromstring_foreign_sep(self):
        a = np.array([1, 2, 3, 4])
        b = np.fromstring('1,2,3,4,', dtype=np.longdouble, sep=',')
        assert_array_equal(a, b)

    def test_fromstring_foreign_value(self):
        with assert_warns(DeprecationWarning):
            b = np.fromstring('1,234', dtype=np.longdouble, sep=' ')
            assert_array_equal(b[0], 1)