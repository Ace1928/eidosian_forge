import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_s_eq_0():
    x = cp.Variable(1)
    s = cp.Variable(1, nonneg=True)
    f = x + 1
    f_recession = x
    obj = cp.perspective(f, s, f_recession=f_recession)
    constr = [-cp.square(x) + 1 >= 0]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()
    assert np.isclose(x.value, -1)
    assert np.isclose(s.value, 0)