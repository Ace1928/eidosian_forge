import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Constraint, TransformationFactory, Var
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
def test_trivial_constraints_skipped(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.c = Constraint(expr=(m.x + m.y) * m.z >= 8)
    m.z.fix(0)
    TransformationFactory('contrib.remove_zero_terms').apply_to(m)
    m.z.unfix()
    self.assertEqual(m.c.lower, 8)
    self.assertIsNone(m.c.upper)
    repn = generate_standard_repn(m.c.body)
    self.assertTrue(repn.is_quadratic())
    self.assertEqual(repn.quadratic_coefs[0], 1)
    self.assertEqual(repn.quadratic_coefs[1], 1)
    self.assertIs(repn.quadratic_vars[0][0], m.x)
    self.assertIs(repn.quadratic_vars[0][1], m.z)
    self.assertIs(repn.quadratic_vars[1][0], m.y)
    self.assertIs(repn.quadratic_vars[1][1], m.z)
    self.assertEqual(repn.constant, 0)