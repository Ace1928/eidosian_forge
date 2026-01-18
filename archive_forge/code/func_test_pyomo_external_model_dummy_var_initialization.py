import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.pyomo_ext_cyipopt import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def test_pyomo_external_model_dummy_var_initialization(self):
    m = pyo.ConcreteModel()
    m.Pin = pyo.Var(initialize=100, bounds=(0, None))
    m.c1 = pyo.Var(initialize=1.0, bounds=(0, None))
    m.c2 = pyo.Var(initialize=1.0, bounds=(0, None))
    m.F = pyo.Var(initialize=10, bounds=(0, None))
    m.P1 = pyo.Var(initialize=75.0)
    m.P2 = pyo.Var(initialize=50.0)
    m.F_con = pyo.Constraint(expr=m.F == 10)
    m.Pin_con = pyo.Constraint(expr=m.Pin == 100)
    m.obj = pyo.Objective(expr=(m.P1 - 90) ** 2 + (m.P2 - 40) ** 2)
    cyipopt_problem = PyomoExternalCyIpoptProblem(m, PressureDropModel(), [m.Pin, m.c1, m.c2, m.F], [m.P1, m.P2])
    expected_dummy_var_value = pyo.value(m.Pin) + pyo.value(m.c1) + pyo.value(m.c2) + pyo.value(m.F) + pyo.value(m.P1) + pyo.value(m.P2)
    self.assertAlmostEqual(pyo.value(m._dummy_variable_CyIpoptPyomoExNLP), expected_dummy_var_value)
    self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body), pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.lower))
    self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body), pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.upper))
    solver = CyIpoptSolver(cyipopt_problem, {'hessian_approximation': 'limited-memory'})
    x, info = solver.solve(tee=False)
    cyipopt_problem.load_x_into_pyomo(x)
    self.assertAlmostEqual(pyo.value(m.c1), 0.1, places=5)
    self.assertAlmostEqual(pyo.value(m.c2), 0.5, places=5)