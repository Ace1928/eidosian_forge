import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.mark.parametrize('n', [2, 3, 11])
def test_psd_tr_square(n):
    ref_s = cp.Variable(nonneg=True)
    ref_P = cp.Variable((n, n), PSD=True)
    obj = cp.quad_over_lin(cp.trace(ref_P), ref_s)
    constraints = [ref_s <= 5, ref_P >> np.eye(n)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.SCS)
    P = cp.Variable((n, n), PSD=True)
    s = cp.Variable(nonneg=True)
    f = cp.perspective(cp.square(cp.trace(P)), s)
    obj = cp.perspective(f, s)
    constraints = [s <= 5, P >> np.eye(n)]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value, atol=0.001)
    assert np.allclose(P.value, ref_P.value, atol=0.0001)