import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_solve_doesnt_converge(self):
    m, _ = make_scalar_model()
    m.x.set_value(30000000000.0)
    solver = pyo.SolverFactory('scipy.secant-newton')
    with self.assertRaisesRegex(RuntimeError, 'Failed to converge'):
        results = solver.solve(m)
    self.assertFalse(solver.converged_with_secant())