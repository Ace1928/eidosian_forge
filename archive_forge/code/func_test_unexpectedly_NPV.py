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
def test_unexpectedly_NPV(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.p = Param(mutable=True, initialize=0)
    e = m.y * cos(m.x / 2)
    e1 = replace_expressions(e, {id(m.x): m.p})
    rep = generate_standard_repn(e1, compute_values=True)
    self.assertEqual(str(rep.to_expression()), 'y')