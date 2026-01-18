import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
def test_val1(self):
    m = ConcreteModel()
    m.v = Var(initialize=2)
    e = inequality(0, m.v, 2)
    self.assertEqual(value(e), True)
    e = inequality(0, m.v, 1)
    self.assertEqual(value(e), False)
    e = inequality(0, m.v, 2, strict=True)
    self.assertEqual(value(e), False)