import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
class TestNDEnumerate:

    def test_ndenumerate_nomasked(self):
        ordinary = np.arange(6.0).reshape((1, 3, 2))
        empty_mask = np.zeros_like(ordinary, dtype=bool)
        with_mask = masked_array(ordinary, mask=empty_mask)
        assert_equal(list(np.ndenumerate(ordinary)), list(ndenumerate(ordinary)))
        assert_equal(list(ndenumerate(ordinary)), list(ndenumerate(with_mask)))
        assert_equal(list(ndenumerate(with_mask)), list(ndenumerate(with_mask, compressed=False)))

    def test_ndenumerate_allmasked(self):
        a = masked_all(())
        b = masked_all((100,))
        c = masked_all((2, 3, 4))
        assert_equal(list(ndenumerate(a)), [])
        assert_equal(list(ndenumerate(b)), [])
        assert_equal(list(ndenumerate(b, compressed=False)), list(zip(np.ndindex((100,)), 100 * [masked])))
        assert_equal(list(ndenumerate(c)), [])
        assert_equal(list(ndenumerate(c, compressed=False)), list(zip(np.ndindex((2, 3, 4)), 2 * 3 * 4 * [masked])))

    def test_ndenumerate_mixedmasked(self):
        a = masked_array(np.arange(12).reshape((3, 4)), mask=[[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
        items = [((1, 2), 6), ((2, 0), 8), ((2, 1), 9), ((2, 2), 10), ((2, 3), 11)]
        assert_equal(list(ndenumerate(a)), items)
        assert_equal(len(list(ndenumerate(a, compressed=False))), a.size)
        for coordinate, value in ndenumerate(a, compressed=False):
            assert_equal(a[coordinate], value)