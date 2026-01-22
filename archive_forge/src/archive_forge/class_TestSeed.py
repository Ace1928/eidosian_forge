import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestSeed:

    def test_scalar(self):
        s = random.RandomState(0)
        assert_equal(s.randint(1000), 684)
        s = random.RandomState(4294967295)
        assert_equal(s.randint(1000), 419)

    def test_array(self):
        s = random.RandomState(range(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState(np.arange(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState([0])
        assert_equal(s.randint(1000), 973)
        s = random.RandomState([4294967295])
        assert_equal(s.randint(1000), 265)

    def test_invalid_scalar(self):
        assert_raises(TypeError, random.RandomState, -0.5)
        assert_raises(ValueError, random.RandomState, -1)

    def test_invalid_array(self):
        assert_raises(TypeError, random.RandomState, [-0.5])
        assert_raises(ValueError, random.RandomState, [-1])
        assert_raises(ValueError, random.RandomState, [4294967296])
        assert_raises(ValueError, random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        assert_raises(ValueError, random.RandomState, np.array([], dtype=np.int64))
        assert_raises(ValueError, random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, random.RandomState, [[1, 2, 3], [4, 5, 6]])

    def test_cannot_seed(self):
        rs = random.RandomState(PCG64(0))
        with assert_raises(TypeError):
            rs.seed(1234)

    def test_invalid_initialization(self):
        assert_raises(ValueError, random.RandomState, MT19937)