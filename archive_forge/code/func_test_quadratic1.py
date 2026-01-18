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
def test_quadratic1(self):
    m = ConcreteModel()
    m.a = Var()
    m.b = Var()
    m.c = Var()
    m.d = Var()
    ab_key = (id(m.a), id(m.b)) if id(m.a) <= id(m.b) else (id(m.b), id(m.a))
    e1 = m.a * 5
    e = e1 * m.b
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {ab_key: 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
        baseline = {(id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])): 5}
    else:
        baseline = {(id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a * m.b
    e = e1 * 5
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {ab_key: 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
        baseline = {(id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])): 5}
    else:
        baseline = {(id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a * m.b
    e = 5 * e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {ab_key: 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
        baseline = {(id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])): 5}
    else:
        baseline = {(id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])): 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    e1 = m.a * 5
    e = m.b * e1
    rep = generate_standard_repn(e)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {ab_key: 5}
    self.assertEqual(baseline, repn_to_dict(rep))
    s = pickle.dumps(rep)
    rep = pickle.loads(s)
    if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
        baseline = {(id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])): 5}
    else:
        baseline = {(id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])): 5}
    self.assertEqual(baseline, repn_to_dict(rep))