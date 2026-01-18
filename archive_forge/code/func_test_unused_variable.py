import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
def test_unused_variable(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.z = pyo.Var(initialize=1.0)
    m.eqn = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 == 1.0)
    variables = [m.x, m.y, m.z]
    expected_hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    hess = get_hessian_of_constraint(m.eqn, variables).toarray()
    np.testing.assert_allclose(hess, expected_hess, rtol=1e-08)