import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.mark.parametrize('n', [2, 3, 11])
def test_psd_mf_persp(n):
    ref_x = cp.Variable(n)
    ref_P = cp.Variable((n, n), PSD=True)
    obj = cp.matrix_frac(ref_x, ref_P)
    constraints = [ref_x == 5, ref_P == np.eye(n)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.SCS)
    x = cp.Variable(n)
    P = cp.Variable((n, n), PSD=True)
    s = cp.Variable(nonneg=True)
    f = cp.matrix_frac(x, P)
    obj = cp.perspective(f, s)
    constraints = [x == 5, P == np.eye(n), s == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value, atol=0.01)
    assert np.allclose(x.value, ref_x.value, atol=0.01)