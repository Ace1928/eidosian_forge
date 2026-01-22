import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
class TestTieCorrect:

    def test_empty(self):
        """An empty array requires no correction, should return 1.0."""
        ranks = np.array([], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_one(self):
        """A single element requires no correction, should return 1.0."""
        ranks = np.array([1.0], dtype=np.float64)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_no_correction(self):
        """Arrays with no ties require no correction."""
        ranks = np.arange(2.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)
        ranks = np.arange(3.0)
        c = tiecorrect(ranks)
        assert_equal(c, 1.0)

    def test_basic(self):
        """Check a few basic examples of the tie correction factor."""
        ranks = np.array([1.0, 2.5, 2.5])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T ** 3 - T) / (N ** 3 - N)
        assert_equal(c, expected)
        ranks = np.array([1.5, 1.5, 3.0])
        c = tiecorrect(ranks)
        T = 2.0
        N = ranks.size
        expected = 1.0 - (T ** 3 - T) / (N ** 3 - N)
        assert_equal(c, expected)
        ranks = np.array([1.0, 3.0, 3.0, 3.0])
        c = tiecorrect(ranks)
        T = 3.0
        N = ranks.size
        expected = 1.0 - (T ** 3 - T) / (N ** 3 - N)
        assert_equal(c, expected)
        ranks = np.array([1.5, 1.5, 4.0, 4.0, 4.0])
        c = tiecorrect(ranks)
        T1 = 2.0
        T2 = 3.0
        N = ranks.size
        expected = 1.0 - (T1 ** 3 - T1 + (T2 ** 3 - T2)) / (N ** 3 - N)
        assert_equal(c, expected)

    def test_overflow(self):
        ntie, k = (2000, 5)
        a = np.repeat(np.arange(k), ntie)
        n = a.size
        out = tiecorrect(rankdata(a))
        assert_equal(out, 1.0 - k * (ntie ** 3 - ntie) / float(n ** 3 - n))