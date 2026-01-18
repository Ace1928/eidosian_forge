import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_power_function_to_string(self):
    m = ConcreteModel()
    m.x = Var()
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(expression_to_string(m.x ** (-3), tc, smap=smap), ('power(x1, (-3))', False))
    self.assertEqual(expression_to_string(m.x ** 0.33, tc, smap=smap), ('x1 ** 0.33', False))
    self.assertEqual(expression_to_string(pow(m.x, 2), tc, smap=smap), ('power(x1, 2)', False))