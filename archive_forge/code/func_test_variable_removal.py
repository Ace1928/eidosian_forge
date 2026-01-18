import pyomo.common.unittest as unittest
from pyomo.opt import (
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_variable_removal(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    opt = pyo.SolverFactory('mosek_persistent')
    opt.set_instance(m)
    self.assertEqual(opt._solver_model.getnumvar(), 2)
    opt.remove_var(m.x)
    self.assertEqual(opt._solver_model.getnumvar(), 1)
    opt.remove_var(m.y)
    self.assertEqual(opt._solver_model.getnumvar(), 0)
    self.assertRaises(ValueError, opt.remove_var, m.x)
    opt.add_var(m.x)
    opt.add_var(m.y)
    self.assertEqual(opt._solver_model.getnumvar(), 2)
    opt.remove_vars(m.x, m.y)
    self.assertEqual(opt._solver_model.getnumvar(), 0)