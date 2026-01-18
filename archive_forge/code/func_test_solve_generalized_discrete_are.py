import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse._sputils import matrix
def test_solve_generalized_discrete_are():
    mat20170120 = _load_data('gendare_20170120_data.npz')
    cases = [(np.array([[0.276923, 0.8234578, 0.950222], [0.04617139, 0.6948286, 0.03444608], [0.09713178, 0.3170995, 0.4387444]]), np.array([[0.3815585, 0.1868726], [0.7655168, 0.4897644], [0.7951999, 0.4455862]]), np.eye(3), np.eye(2), np.array([[0.646313, 0.2760251, 0.1626117], [0.7093648, 0.6797027, 0.1189977], [0.7546867, 0.655098, 0.4983641]]), np.zeros((3, 2)), None), (np.array([[0.276923, 0.8234578, 0.950222], [0.04617139, 0.6948286, 0.03444608], [0.09713178, 0.3170995, 0.4387444]]), np.array([[0.3815585, 0.1868726], [0.7655168, 0.4897644], [0.7951999, 0.4455862]]), np.eye(3), np.eye(2), np.array([[0.646313, 0.2760251, 0.1626117], [0.7093648, 0.6797027, 0.1189977], [0.7546867, 0.655098, 0.4983641]]), np.ones((3, 2)), None), (mat20170120['A'], mat20170120['B'], mat20170120['Q'], mat20170120['R'], None, mat20170120['S'], None)]
    max_atol = (1.5e-11, 1.5e-11, 3.5e-16)

    def _test_factory(case, atol):
        """Checks if X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) is true"""
        a, b, q, r, e, s, knownfailure = case
        if knownfailure:
            pytest.xfail(reason=knownfailure)
        x = solve_discrete_are(a, b, q, r, e, s)
        if e is None:
            e = np.eye(a.shape[0])
        if s is None:
            s = np.zeros_like(b)
        res = a.conj().T.dot(x.dot(a)) - e.conj().T.dot(x.dot(e)) + q
        res -= (a.conj().T.dot(x.dot(b)) + s).dot(solve(r + b.conj().T.dot(x.dot(b)), b.conj().T.dot(x.dot(a)) + s.conj().T))
        assert_allclose(res, np.zeros_like(res), atol=atol)
    for ind, case in enumerate(cases):
        _test_factory(case, max_atol[ind])