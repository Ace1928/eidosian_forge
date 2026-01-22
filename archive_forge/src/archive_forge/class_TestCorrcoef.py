import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
class TestCorrcoef:

    def setup_method(self):
        self.data = array(np.random.rand(12))
        self.data2 = array(np.random.rand(12))

    def test_ddof(self):
        x, y = (self.data, self.data2)
        expected = np.corrcoef(x)
        expected2 = np.corrcoef(x, y)
        with suppress_warnings() as sup:
            warnings.simplefilter('always')
            assert_warns(DeprecationWarning, corrcoef, x, ddof=-1)
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(np.corrcoef(x, ddof=0), corrcoef(x, ddof=0))
            assert_almost_equal(corrcoef(x, ddof=-1), expected)
            assert_almost_equal(corrcoef(x, y, ddof=-1), expected2)
            assert_almost_equal(corrcoef(x, ddof=3), expected)
            assert_almost_equal(corrcoef(x, y, ddof=3), expected2)

    def test_bias(self):
        x, y = (self.data, self.data2)
        expected = np.corrcoef(x)
        with suppress_warnings() as sup:
            warnings.simplefilter('always')
            assert_warns(DeprecationWarning, corrcoef, x, y, True, False)
            assert_warns(DeprecationWarning, corrcoef, x, y, True, True)
            assert_warns(DeprecationWarning, corrcoef, x, bias=False)
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(corrcoef(x, bias=1), expected)

    def test_1d_without_missing(self):
        x = self.data
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False), corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True), corrcoef(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False), corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True), corrcoef(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.corrcoef(nx), corrcoef(x))
        assert_almost_equal(np.corrcoef(nx, rowvar=False), corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(np.corrcoef(nx, rowvar=False, bias=True), corrcoef(x, rowvar=False, bias=True))
        try:
            corrcoef(x, allow_masked=False)
        except ValueError:
            pass
        nx = x[1:-1]
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1]))
        assert_almost_equal(np.corrcoef(nx, nx[::-1], rowvar=False), corrcoef(x, x[::-1], rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1], bias=1))
            assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1], ddof=2))

    def test_2d_with_missing(self):
        x = self.data
        x[-1] = masked
        x = x.reshape(3, 4)
        test = corrcoef(x)
        control = np.corrcoef(x)
        assert_almost_equal(test[:-1, :-1], control[:-1, :-1])
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'bias and ddof have no effect')
            assert_almost_equal(corrcoef(x, ddof=-2)[:-1, :-1], control[:-1, :-1])
            assert_almost_equal(corrcoef(x, ddof=3)[:-1, :-1], control[:-1, :-1])
            assert_almost_equal(corrcoef(x, bias=1)[:-1, :-1], control[:-1, :-1])