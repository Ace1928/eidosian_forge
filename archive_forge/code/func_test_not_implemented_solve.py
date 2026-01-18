import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_not_implemented_solve(self):
    m, nlp = make_simple_model()
    solver = SquareNlpSolverBase(nlp)
    msg = 'has not implemented the solve method'
    with self.assertRaisesRegex(NotImplementedError, msg):
        solver.solve()