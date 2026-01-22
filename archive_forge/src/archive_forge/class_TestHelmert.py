import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestHelmert:

    def test_orthogonality(self):
        for n in range(1, 7):
            H = helmert(n, full=True)
            Id = np.eye(n)
            assert_allclose(H.dot(H.T), Id, atol=1e-12)
            assert_allclose(H.T.dot(H), Id, atol=1e-12)

    def test_subspace(self):
        for n in range(2, 7):
            H_full = helmert(n, full=True)
            H_partial = helmert(n)
            for U in (H_full[1:, :].T, H_partial.T):
                C = np.eye(n) - np.full((n, n), 1 / n)
                assert_allclose(U.dot(U.T), C)
                assert_allclose(U.T.dot(U), np.eye(n - 1), atol=1e-12)