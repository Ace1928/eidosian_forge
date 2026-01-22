import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
class TestLogM:

    def test_nils(self):
        a = array([[-2.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -3.0, 10.0, 3.0, 3.0, 3.0, 0.0], [0.0, 0.0, 2.0, 15.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 15.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0, 10.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 25.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0]])
        m = identity(7) * 3.1 + 0j - a
        logm(m, disp=False)

    def test_al_mohy_higham_2012_experiment_1_logm(self):
        A = _get_al_mohy_higham_2012_experiment_1()
        A_logm, info = logm(A, disp=False)
        A_round_trip = expm(A_logm)
        assert_allclose(A_round_trip, A, rtol=5e-05, atol=1e-14)

    def test_al_mohy_higham_2012_experiment_1_funm_log(self):
        A = _get_al_mohy_higham_2012_experiment_1()
        A_funm_log, info = funm(A, np.log, disp=False)
        A_round_trip = expm(A_funm_log)
        assert_(not np.allclose(A_round_trip, A, rtol=1e-05, atol=1e-14))

    def test_round_trip_random_float(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                W = np.linalg.eigvals(M)
                err_msg = f'M:{M} eivals:{W}'
                M_sqrtm, info = sqrtm(M, disp=False)
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                assert_allclose(M_sqrtm_round_trip, M)
                M_logm, info = logm(M, disp=False)
                M_logm_round_trip = expm(M_logm)
                assert_allclose(M_logm_round_trip, M, err_msg=err_msg)

    def test_round_trip_random_complex(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                M_logm, info = logm(M, disp=False)
                M_round_trip = expm(M_logm)
                assert_allclose(M_round_trip, M)

    def test_logm_type_preservation_and_conversion(self):
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in ([[1, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 1], [1, 1]], [[2, 3], [1, 2]]):
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any((w.imag or w.real < 0 for w in W)))
            A = np.array(matrix_as_list, dtype=float)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char not in complex_dtype_chars)
            A = np.array(matrix_as_list, dtype=complex)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char in complex_dtype_chars)
            A = -np.array(matrix_as_list, dtype=float)
            A_logm, info = logm(A, disp=False)
            assert_(A_logm.dtype.char in complex_dtype_chars)

    def test_complex_spectrum_real_logm(self):
        M = [[1, 1, 2], [2, 1, 1], [1, 2, 1]]
        for dt in (float, complex):
            X = np.array(M, dtype=dt)
            w = scipy.linalg.eigvals(X)
            assert_(0.01 < np.absolute(w.imag).sum())
            Y, info = logm(X, disp=False)
            assert_(np.issubdtype(Y.dtype, np.inexact))
            assert_allclose(expm(Y), X)

    def test_real_mixed_sign_spectrum(self):
        for M in ([[1, 0], [0, -1]], [[0, 1], [1, 0]]):
            for dt in (float, complex):
                A = np.array(M, dtype=dt)
                A_logm, info = logm(A, disp=False)
                assert_(np.issubdtype(A_logm.dtype, np.complexfloating))

    def test_exactly_singular(self):
        A = np.array([[0, 0], [1j, 1j]])
        B = np.asarray([[1, 1], [0, 0]])
        for M in (A, A.T, B, B.T):
            expected_warning = _matfuncs_inv_ssq.LogmExactlySingularWarning
            L, info = assert_warns(expected_warning, logm, M, disp=False)
            E = expm(L)
            assert_allclose(E, M, atol=1e-14)

    def test_nearly_singular(self):
        M = np.array([[1e-100]])
        expected_warning = _matfuncs_inv_ssq.LogmNearlySingularWarning
        L, info = assert_warns(expected_warning, logm, M, disp=False)
        E = expm(L)
        assert_allclose(E, M, atol=1e-14)

    def test_opposite_sign_complex_eigenvalues(self):
        E = [[0, 1], [-1, 0]]
        L = [[0, np.pi * 0.5], [-np.pi * 0.5, 0]]
        assert_allclose(expm(L), E, atol=1e-14)
        assert_allclose(logm(E), L, atol=1e-14)
        E = [[1j, 4], [0, -1j]]
        L = [[1j * np.pi * 0.5, 2 * np.pi], [0, -1j * np.pi * 0.5]]
        assert_allclose(expm(L), E, atol=1e-14)
        assert_allclose(logm(E), L, atol=1e-14)
        E = [[1j, 0], [0, -1j]]
        L = [[1j * np.pi * 0.5, 0], [0, -1j * np.pi * 0.5]]
        assert_allclose(expm(L), E, atol=1e-14)
        assert_allclose(logm(E), L, atol=1e-14)