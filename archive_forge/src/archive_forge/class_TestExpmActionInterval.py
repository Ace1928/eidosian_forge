from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
class TestExpmActionInterval:

    def test_sparse_expm_multiply_interval(self):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        n = 40
        k = 3
        endpoint = True
        for num in (14, 13, 2):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            v = np.random.randn(n)
            for target in (B, v):
                X = expm_multiply(A, target, start=start, stop=stop, num=num, endpoint=endpoint)
                samples = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint)
                with suppress_warnings() as sup:
                    sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
                    sup.filter(SparseEfficiencyWarning, 'spsolve is more efficient when sparse b is in the CSC matrix format')
                    for solution, t in zip(X, samples):
                        assert_allclose(solution, sp_expm(t * A).dot(target))

    def test_expm_multiply_interval_vector(self):
        np.random.seed(1234)
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        for num, n in product([14, 13, 2], [1, 2, 5, 20, 40]):
            A = scipy.linalg.inv(np.random.randn(n, n))
            v = np.random.randn(n)
            samples = np.linspace(num=num, **interval)
            X = expm_multiply(A, v, num=num, **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(v))
            Xguess = estimated(expm_multiply)(aslinearoperator(A), v, num=num, **interval)
            Xgiven = expm_multiply(aslinearoperator(A), v, num=num, **interval, traceA=np.trace(A))
            Xwrong = expm_multiply(aslinearoperator(A), v, num=num, **interval, traceA=np.trace(A) * 5)
            for sol_guess, sol_given, sol_wrong, t in zip(Xguess, Xgiven, Xwrong, samples):
                correct = sp_expm(t * A).dot(v)
                assert_allclose(sol_guess, correct)
                assert_allclose(sol_given, correct)
                assert_allclose(sol_wrong, correct)

    def test_expm_multiply_interval_matrix(self):
        np.random.seed(1234)
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        for num, n, k in product([14, 13, 2], [1, 2, 5, 20, 40], [1, 2]):
            A = scipy.linalg.inv(np.random.randn(n, n))
            B = np.random.randn(n, k)
            samples = np.linspace(num=num, **interval)
            X = expm_multiply(A, B, num=num, **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(B))
            X = estimated(expm_multiply)(aslinearoperator(A), B, num=num, **interval)
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(B))

    def test_sparse_expm_multiply_interval_dtypes(self):
        A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))
        A = scipy.sparse.diags(-1j * np.arange(5), format='csr', dtype=complex)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(-1j * np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))
        A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
        B = np.full(5, 1j, dtype=complex)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))

    def test_expm_multiply_interval_status_0(self):
        self._help_test_specific_expm_interval_status(0)

    def test_expm_multiply_interval_status_1(self):
        self._help_test_specific_expm_interval_status(1)

    def test_expm_multiply_interval_status_2(self):
        self._help_test_specific_expm_interval_status(2)

    def _help_test_specific_expm_interval_status(self, target_status):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        num = 13
        endpoint = True
        n = 5
        k = 2
        nrepeats = 10
        nsuccesses = 0
        for num in [14, 13, 2] * nrepeats:
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            status = _expm_multiply_interval(A, B, start=start, stop=stop, num=num, endpoint=endpoint, status_only=True)
            if status == target_status:
                X, status = _expm_multiply_interval(A, B, start=start, stop=stop, num=num, endpoint=endpoint, status_only=False)
                assert_equal(X.shape, (num, n, k))
                samples = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint)
                for solution, t in zip(X, samples):
                    assert_allclose(solution, sp_expm(t * A).dot(B))
                nsuccesses += 1
        if not nsuccesses:
            msg = 'failed to find a status-' + str(target_status) + ' interval'
            raise Exception(msg)