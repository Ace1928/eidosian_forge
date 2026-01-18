import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def test_pow_of_lin_sum(self):
    m = ConcreteModel()
    m.x = Var(range(4))
    e = sum((x for x in m.x.values())) ** 2
    rep = generate_standard_repn(e, compute_values=False, quadratic=False)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), None)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertFalse(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 4)
    baseline = {}
    self.assertEqual(baseline, repn_to_dict(rep))
    rep = generate_standard_repn(e, compute_values=False, quadratic=True)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 10)
    self.assertTrue(len(rep.quadratic_coefs) == 10)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {(id(i), id(j)): 2 for i in m.x.values() for j in m.x.values() if id(i) < id(j)}
    baseline.update({(id(i), id(i)): 1 for i in m.x.values()})
    self.assertEqual(baseline, repn_to_dict(rep))