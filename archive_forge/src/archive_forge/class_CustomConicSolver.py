import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
class CustomConicSolver(SCS):

    def name(self) -> str:
        return 'CUSTOM_CONIC_SOLVER'

    def solve_via_data(self, *args, **kwargs):
        raise CustomConicSolverCalled()