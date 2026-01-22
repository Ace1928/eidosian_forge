import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class ImplicitFunction1(object):

    def __init__(self):
        self._model = self._make_model()

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2, 3])
        m.J = pyo.Set(initialize=[1, 2])
        m.x = pyo.Var(m.I, initialize=1.0)
        m.p = pyo.Var(m.J, initialize=1.0)
        m.con1 = pyo.Constraint(expr=m.x[2] ** 2 + m.x[3] ** 2 == m.p[1])
        m.con2 = pyo.Constraint(expr=2 * m.x[1] + 3 * m.x[2] - 4 * m.x[3] == m.p[1] ** 2 - m.p[2])
        m.con3 = pyo.Constraint(expr=m.p[2] ** 1.5 == 2 * pyo.exp(m.x[2] / m.x[3]))
        m.obj = pyo.Objective(expr=0.0)
        return m

    def get_parameters(self):
        m = self._model
        return [m.p[1], m.p[2]]

    def get_variables(self):
        m = self._model
        return [m.x[1], m.x[2], m.x[3]]

    def get_equations(self):
        m = self._model
        return [m.con1, m.con2, m.con3]

    def get_input_output_sequence(self):
        p1_inputs = [1.0, 2.0, 3.0]
        p2_inputs = [1.0, 2.0, 3.0]
        inputs = list(itertools.product(p1_inputs, p2_inputs))
        outputs = [(2.498253, -0.569676, 0.821869), (0.89853, 0.327465, 0.944863), (-0.589294, 0.690561, 0.723274), (5.033063, -0.805644, 1.162299), (2.97782, 0.463105, 1.336239), (1.080826, 0.976601, 1.022864), (8.327101, -0.986708, 1.423519), (5.922325, 0.567186, 1.636551), (3.711364, 1.196087, 1.252747)]
        return list(zip(inputs, outputs))