import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.mark.parametrize('p', [1, 2])
def test_p_norms(p):
    x = cp.Variable(3)
    s = cp.Variable(nonneg=True, name='s')
    f = cp.norm(x, p)
    obj = cp.perspective(f, s)
    constraints = [1 == s, x >= [1, 2, 3]]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)
    ref_x = cp.Variable(3, pos=True)
    ref_s = cp.Variable(pos=True)
    obj = cp.sum(cp.power(ref_x, p) / cp.power(ref_s, p - 1))
    ref_constraints = [ref_x >= [1, 2, 3], ref_s == 1]
    ref_prob = cp.Problem(cp.Minimize(obj), ref_constraints)
    ref_prob.solve(gp=True)
    assert np.isclose(prob.value ** p, ref_prob.value)
    assert np.allclose(x.value, ref_x.value)
    if p != 1:
        assert np.isclose(s.value, ref_s.value)