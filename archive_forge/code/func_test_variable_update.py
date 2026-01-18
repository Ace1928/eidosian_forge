import pyomo.common.unittest as unittest
from pyomo.opt import (
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_variable_update(self):
    """
        Test based on milo1.py problem from MOSEK documentation.
        """
    cont_sol_to_get = [1.948, 4.922]
    int_sol_to_get = [5, 0]
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c1 = pyo.Constraint(expr=50 * m.x + 31 * m.y <= 250)
    m.c2 = pyo.Constraint(expr=3 * m.x - 2 * m.y >= -4)
    m.o = pyo.Objective(expr=m.x + 0.64 * m.y, sense=pyo.maximize)
    opt = pyo.SolverFactory('mosek_persistent')
    opt.set_instance(m)
    opt.solve(m)
    self.assertAlmostEqual(m.x.value, cont_sol_to_get[0], places=2)
    self.assertAlmostEqual(m.y.value, cont_sol_to_get[1], places=2)
    m.x.setlb = 0
    m.x.setub = None
    m.x.domain = pyo.Integers
    m.y.setlb = 0
    m.y.setub = None
    m.y.domain = pyo.Integers
    m.z = pyo.Var()
    opt.update_vars(m.x, m.y)
    self.assertRaises(ValueError, opt.update_var, m.z)
    opt.add_var(m.z)
    opt.solve(m)
    self.assertAlmostEqual(m.x.value, int_sol_to_get[0], places=1)
    self.assertAlmostEqual(m.y.value, int_sol_to_get[1], places=1)