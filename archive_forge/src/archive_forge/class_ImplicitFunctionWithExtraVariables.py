import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class ImplicitFunctionWithExtraVariables(ImplicitFunction1):
    """This is the same system as ImplicitFunction1, but now some
    of the hand-coded constants have been replaced by unfixed variables.
    These variables will be completely ignored and treated as constants
    by the implicit functions.

    """

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2, 3])
        m.J = pyo.Set(initialize=[1, 2])
        m.K = pyo.Set(initialize=[1, 2, 3])
        m.x = pyo.Var(m.I, initialize=1.0)
        m.p = pyo.Var(m.J, initialize=1.0)
        m.const = pyo.Var(m.K, initialize=1.0)
        m.const[1].set_value(1.0)
        m.const[2].set_value(2.0)
        m.const[3].set_value(1.5)
        m.con1 = pyo.Constraint(expr=m.const[1] * m.x[2] ** 2 + m.x[3] ** 2 == m.p[1])
        m.con2 = pyo.Constraint(expr=m.const[2] * m.x[1] + 3 * m.x[2] - 4 * m.x[3] == m.p[1] ** 2 - m.p[2])
        m.con3 = pyo.Constraint(expr=m.p[2] ** m.const[3] == 2 * pyo.exp(m.x[2] / m.x[3]))
        m.obj = pyo.Objective(expr=0.0)
        return m