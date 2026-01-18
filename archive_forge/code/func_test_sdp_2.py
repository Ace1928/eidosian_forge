import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_sdp_2(solver, places: int=3, duals: bool=True, **kwargs) -> SolverTestHelper:
    sth = sdp_2()
    sth.solve(solver, **kwargs)
    sth.verify_objective(places)
    sth.check_primal_feasibility(places)
    sth.verify_primal_values(places)
    if duals:
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
    return sth