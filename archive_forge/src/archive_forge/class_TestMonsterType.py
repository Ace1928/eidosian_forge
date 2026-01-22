import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestMonsterType:
    """Test deeply nested subtypes."""

    def test1(self):
        simple1 = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', int), ('ye', simple1), ('yi', np.dtype((int, (3, 2))))])
        b = np.dtype([('yo', int), ('ye', simple1), ('yi', np.dtype((int, (3, 2))))])
        assert_dtype_equal(a, b)
        c = np.dtype([('yo', int), ('ye', simple1), ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', int), ('ye', simple1), ('yi', np.dtype((a, (3, 2))))])
        assert_dtype_equal(c, d)

    @pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
    def test_list_recursion(self):
        l = list()
        l.append(('f', l))
        with pytest.raises(RecursionError):
            np.dtype(l)

    @pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
    def test_tuple_recursion(self):
        d = np.int32
        for i in range(100000):
            d = (d, (1,))
        with pytest.raises(RecursionError):
            np.dtype(d)

    @pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
    def test_dict_recursion(self):
        d = dict(names=['self'], formats=[None], offsets=[0])
        d['formats'][0] = d
        with pytest.raises(RecursionError):
            np.dtype(d)