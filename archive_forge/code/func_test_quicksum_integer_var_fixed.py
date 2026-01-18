import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_quicksum_integer_var_fixed(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var(domain=Binary)
    m.c = Constraint(expr=quicksum([m.y, m.y], linear=True) == 1)
    m.o = Objective(expr=m.x ** 2)
    m.y.fix(1)
    outs = StringIO()
    m.write(outs, format='gams')
    self.assertIn('USING nlp', outs.getvalue())