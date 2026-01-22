import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
def _make_model(self):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3])
    m.J = pyo.Set(initialize=[1, 2])
    m.x = pyo.Var(m.I, initialize=1.0)
    m.p = pyo.Var(m.J, initialize=1.0)
    m.con1 = pyo.Constraint(expr=m.x[2] ** 2 + m.x[3] ** 2 == 1.0)
    m.con2 = pyo.Constraint(expr=2 * m.x[1] + 3 * m.x[2] - 4 * m.x[3] == 0.0)
    m.con3 = pyo.Constraint(expr=1.0 == 2 * pyo.exp(m.x[2] / m.x[3]))
    m.obj = pyo.Objective(expr=0.0)
    return m