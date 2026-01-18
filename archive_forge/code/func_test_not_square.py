import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_not_square(self):
    m, _ = make_simple_model()
    m.con4 = pyo.Constraint(expr=m.x[1] == m.x[2])
    nlp = PyomoNLP(m)
    msg = 'same numbers of variables as equality constraints'
    with self.assertRaisesRegex(RuntimeError, msg):
        solver = SquareNlpSolverBase(nlp)