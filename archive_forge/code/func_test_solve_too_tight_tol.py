import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_solve_too_tight_tol(self):
    m, _ = make_simple_model()
    solver = pyo.SolverFactory('scipy.fsolve', options=dict(xtol=0.001, maxfev=20, tol=1e-08))
    msg = 'does not satisfy the function tolerance'
    with self.assertRaisesRegex(RuntimeError, msg):
        res = solver.solve(m)