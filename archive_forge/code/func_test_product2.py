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
def test_product2(self):
    m = ConcreteModel()
    m.v = Var(initialize=2)
    m.w = Var(initialize=3)
    m.v.fixed = True
    m.w.fixed = True
    e = m.v * m.w
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '6')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'v*w')
    e = m.w * m.v
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '6')
    m.v.value = 0
    e = m.v * m.w
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '0')
    rep = generate_standard_repn(e, compute_values=False)
    self.assertEqual(str(rep.to_expression()), 'v*w')
    e = m.w * m.v
    m.w.fixed = False
    rep = generate_standard_repn(e, compute_values=True)
    self.assertEqual(str(rep.to_expression()), '0')