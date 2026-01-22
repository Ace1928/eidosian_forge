import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestFromiter:

    def makegen(self):
        return (x ** 2 for x in range(24))

    def test_types(self):
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    def test_lengths(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)

    def test_values(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(np.all(a == expected, axis=0))
        assert_(np.all(a20 == expected[:20], axis=0))

    def load_data(self, n, eindex):
        for e in range(n):
            if e == eindex:
                raise NIterError('error at index %s' % eindex)
            yield e

    @pytest.mark.parametrize('dtype', [int, object])
    @pytest.mark.parametrize(['count', 'error_index'], [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    @pytest.mark.parametrize('dtype', ['S', 'S0', 'V0', 'U0'])
    def test_empty_not_structured(self, dtype):
        with pytest.raises(ValueError, match='Must specify length'):
            np.fromiter([], dtype=dtype)

    @pytest.mark.parametrize(['dtype', 'data'], [('d', [1, 2, 3, 4, 5, 6, 7, 8, 9]), ('O', [1, 2, 3, 4, 5, 6, 7, 8, 9]), ('i,O', [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]), ('2i', [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]), (np.dtype(('O', (2, 3))), [((1, 2, 3), (3, 4, 5)), ((3, 2, 1), (5, 4, 3))])])
    @pytest.mark.parametrize('length_hint', [0, 1])
    def test_growth_and_complicated_dtypes(self, dtype, data, length_hint):
        dtype = np.dtype(dtype)
        data = data * 100

        class MyIter:

            def __length_hint__(self):
                return length_hint

            def __iter__(self):
                return iter(data)
        res = np.fromiter(MyIter(), dtype=dtype)
        expected = np.array(data, dtype=dtype)
        assert_array_equal(res, expected)

    def test_empty_result(self):

        class MyIter:

            def __length_hint__(self):
                return 10

            def __iter__(self):
                return iter([])
        res = np.fromiter(MyIter(), dtype='d')
        assert res.shape == (0,)
        assert res.dtype == 'd'

    def test_too_few_items(self):
        msg = 'iterator too short: Expected 10 but iterator had only 3 items.'
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    def test_failed_itemsetting(self):
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))