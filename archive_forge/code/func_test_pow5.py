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
def test_pow5(self):
    m = ConcreteModel()
    m.a = Var(initialize=2)
    m.b = Var(initialize=2)
    e = sin(m.a) ** 2
    rep = generate_standard_repn(e, compute_values=False, quadratic=True)
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
    self.assertTrue(len(rep.nonlinear_vars) == 1)
    baseline = {}
    self.assertEqual(baseline, repn_to_dict(rep))
    e = (m.a ** 2) ** 2
    rep = generate_standard_repn(e, compute_values=False, quadratic=True)
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
    self.assertTrue(len(rep.nonlinear_vars) == 1)
    baseline = {}
    self.assertEqual(baseline, repn_to_dict(rep))
    e = (m.a + m.b) ** 2
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
    self.assertTrue(len(rep.nonlinear_vars) == 2)
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
    self.assertTrue(len(rep.quadratic_vars) == 3)
    self.assertTrue(len(rep.quadratic_coefs) == 3)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {(id(m.a), id(m.a)): 1, (id(m.b), id(m.b)): 1}
    if id(m.a) < id(m.b):
        baseline[id(m.a), id(m.b)] = 2
    else:
        baseline[id(m.b), id(m.a)] = 2
    self.assertEqual(baseline, repn_to_dict(rep))
    e = (m.a + 3) ** 2
    rep = generate_standard_repn(e, compute_values=False, quadratic=True)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 1)
    self.assertTrue(len(rep.linear_coefs) == 1)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 9, id(m.a): 6, (id(m.a), id(m.a)): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    rep = generate_standard_repn(e, compute_values=True, quadratic=True)
    self.assertFalse(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 2)
    self.assertFalse(rep.is_constant())
    self.assertFalse(rep.is_linear())
    self.assertTrue(rep.is_quadratic())
    self.assertTrue(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 1)
    self.assertTrue(len(rep.linear_coefs) == 1)
    self.assertTrue(len(rep.quadratic_vars) == 1)
    self.assertTrue(len(rep.quadratic_coefs) == 1)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 9, id(m.a): 6, (id(m.a), id(m.a)): 1}
    self.assertEqual(baseline, repn_to_dict(rep))
    m.a.fixed = True
    rep = generate_standard_repn(e, compute_values=True, quadratic=True)
    self.assertTrue(rep.is_fixed())
    self.assertEqual(rep.polynomial_degree(), 0)
    self.assertTrue(rep.is_constant())
    self.assertTrue(rep.is_linear())
    self.assertFalse(rep.is_quadratic())
    self.assertFalse(rep.is_nonlinear())
    self.assertTrue(len(rep.linear_vars) == 0)
    self.assertTrue(len(rep.linear_coefs) == 0)
    self.assertTrue(len(rep.quadratic_vars) == 0)
    self.assertTrue(len(rep.quadratic_coefs) == 0)
    self.assertTrue(rep.nonlinear_expr is None)
    self.assertTrue(len(rep.nonlinear_vars) == 0)
    baseline = {None: 25}
    self.assertEqual(baseline, repn_to_dict(rep))