import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_quicksum(self):
    m = ConcreteModel()
    m.y = Var(domain=Binary)
    m.c = Constraint(expr=quicksum([m.y, m.y], linear=True) == 1)
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(('x1 + x1', False), expression_to_string(m.c.body, tc, smap=smap))
    m.x = Var()
    m.c2 = Constraint(expr=quicksum([m.x, m.y], linear=True) == 1)
    self.assertEqual(('x2 + x1', False), expression_to_string(m.c2.body, tc, smap=smap))
    m.y.fix(1)
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(('1 + 1', False), expression_to_string(m.c.body, tc, smap=smap))
    m.x = Var()
    m.c2 = Constraint(expr=quicksum([m.x, m.y], linear=True) == 1)
    self.assertEqual(('x1 + 1', False), expression_to_string(m.c2.body, tc, smap=smap))