import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
def test_reclassify_deactivated_disjuncts(self):
    m = ConcreteModel()
    m.d = Disjunct([1, 2, 3])
    m.disjunction = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
    m.d[1].deactivate()
    m.d[2].indicator_var = True
    m.d[3].indicator_var = False
    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    self.assertTrue(m.d[1].indicator_var.fixed)
    self.assertFalse(value(m.d[1].indicator_var))
    self.assertFalse(m.d[1].active)
    self.assertEqual(m.d[1].ctype, Block)
    self.assertTrue(m.d[2].indicator_var.fixed)
    self.assertTrue(value(m.d[2].indicator_var))
    self.assertTrue(m.d[2].active)
    self.assertTrue(m.d[3].indicator_var.fixed)
    self.assertFalse(value(m.d[3].indicator_var))
    self.assertFalse(m.d[3].active)
    self.assertEqual(m.d[1].ctype, Block)
    self.assertEqual(m.d[2].ctype, Block)