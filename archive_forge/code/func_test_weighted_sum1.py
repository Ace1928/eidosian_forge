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
def test_weighted_sum1(self):
    m = ConcreteModel()
    m.a = Var()
    m.b = Var()
    m.c = Var()
    m.d = Var()
    e1 = m.a + m.b
    e = e1 * 5
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 5, id(m.b): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 5, id(rep.linear_vars[1]): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e = 5 * e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 5, id(m.b): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 5, id(rep.linear_vars[1]): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a + m.b
    e = 5 * 2 * e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 10, id(m.b): 10}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 10, id(rep.linear_vars[1]): 10}
    self.assertEqual(baseline, repn_to_dict(rep))
    e = 5 * (m.a + 2 * (m.a + m.b))
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 1)
    self.assertFalse(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 2)
    self.assertTrue(len(rep.linear_coefs) == 2)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {id(m.a): 15, id(m.b): 10}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    baseline = {id(rep.linear_vars[0]): 15, id(rep.linear_vars[1]): 10}
    self.assertEqual(baseline, repn_to_dict(rep))