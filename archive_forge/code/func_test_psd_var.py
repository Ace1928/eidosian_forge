import math
import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.tests.base_test import BaseTest
def test_psd_var(self) -> None:
    """Test PSD variable.
        """
    s = cp.Variable((2, 2), PSD=True)
    var_dict = {s.id: s}
    obj = cp.Maximize(cp.minimum(s[0, 1], 10))
    const = [cp.diag(s) == np.ones(2)]
    problem = cp.Problem(obj, const)
    data, _, _ = problem.get_problem_data(solver=cp.SCS)
    param_cone_prog = data[cp.settings.PARAM_PROB]
    solver = SCS()
    raw_solution = solver.solve_via_data(data, warm_start=False, verbose=False, solver_opts={})['x']
    sltn_dict = param_cone_prog.split_solution(raw_solution, active_vars=var_dict)
    self.assertEqual(sltn_dict[s.id].shape, s.shape)
    sltn_value = sltn_dict[s.id]
    adjoint = param_cone_prog.split_adjoint(sltn_dict)
    self.assertEqual(adjoint.shape, raw_solution.shape)
    self.assertTrue(any(sltn_value[0, 0] == adjoint))
    self.assertTrue(any(sltn_value[1, 1] == adjoint))
    self.assertTrue(any(np.isclose(2 * sltn_value[0, 1], adjoint)))
    self.assertTrue(any(np.isclose(2 * sltn_value[1, 0], adjoint)))
    problem.solve(solver=cp.SCS, eps=1e-05)
    self.assertItemsAlmostEqual(s.value, sltn_value)