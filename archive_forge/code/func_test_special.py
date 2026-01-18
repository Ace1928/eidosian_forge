import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_special(self):
    for inf in [np.inf, -np.inf]:
        a = np.array([[inf, np.nan], [np.nan, np.nan]])
        a = np.ma.masked_array(a, mask=np.isnan(a))
        assert_equal(np.ma.median(a, axis=0), [inf, np.nan])
        assert_equal(np.ma.median(a, axis=1), [inf, np.nan])
        assert_equal(np.ma.median(a), inf)
        a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
        a = np.ma.masked_array(a, mask=np.isnan(a))
        assert_array_equal(np.ma.median(a, axis=1), inf)
        assert_array_equal(np.ma.median(a, axis=1).mask, False)
        assert_array_equal(np.ma.median(a, axis=0), a[0])
        assert_array_equal(np.ma.median(a), inf)
        a = np.array([[inf, inf], [inf, inf]])
        assert_equal(np.ma.median(a), inf)
        assert_equal(np.ma.median(a, axis=0), inf)
        assert_equal(np.ma.median(a, axis=1), inf)
        a = np.array([[inf, 7, -inf, -9], [-10, np.nan, np.nan, 5], [4, np.nan, np.nan, inf]], dtype=np.float32)
        a = np.ma.masked_array(a, mask=np.isnan(a))
        if inf > 0:
            assert_equal(np.ma.median(a, axis=0), [4.0, 7.0, -inf, 5.0])
            assert_equal(np.ma.median(a), 4.5)
        else:
            assert_equal(np.ma.median(a, axis=0), [-10.0, 7.0, -inf, -9.0])
            assert_equal(np.ma.median(a), -2.5)
        assert_equal(np.ma.median(a, axis=1), [-1.0, -2.5, inf])
        for i in range(0, 10):
            for j in range(1, 10):
                a = np.array([[np.nan] * i + [inf] * j] * 2)
                a = np.ma.masked_array(a, mask=np.isnan(a))
                assert_equal(np.ma.median(a), inf)
                assert_equal(np.ma.median(a, axis=1), inf)
                assert_equal(np.ma.median(a, axis=0), [np.nan] * i + [inf] * j)