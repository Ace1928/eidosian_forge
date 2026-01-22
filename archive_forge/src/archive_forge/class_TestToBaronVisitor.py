import os
from filecmp import cmp
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
from pyomo.common.fileutils import this_file_dir
import pyomo.core.expr as EXPR
from pyomo.core.base import SymbolMap
from pyomo.environ import (
from pyomo.repn.plugins.baron_writer import expression_to_string
class TestToBaronVisitor(unittest.TestCase):

    def test_pow(self):
        variables = OrderedSet()
        smap = SymbolMap()
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.y = Var(initialize=2)
        m.p = Param(mutable=True, initialize=0)
        e = m.x ** m.y
        test = expression_to_string(e, variables, smap)
        self.assertEqual(test, 'exp((x) * log(y))')
        e = m.x ** (3 + EXPR.ProductExpression((m.p, m.y)))
        test = expression_to_string(e, variables, smap)
        self.assertEqual(test, 'x ^ 3')
        e = (3 + EXPR.ProductExpression((m.p, m.y))) ** m.x
        test = expression_to_string(e, variables, smap)
        self.assertEqual(test, '3 ^ x')

    def test_issue_2819(self):
        m = ConcreteModel()
        m.x = Var()
        m.z = Var()
        t = 0.55
        m.x.fix(3.5)
        e = (m.x - 4) ** 2 + (m.z - 1) ** 2 - t
        variables = OrderedSet()
        smap = SymbolMap()
        test = expression_to_string(e, variables, smap)
        self.assertEqual(test, '(-0.5) ^ 2 + (z - 1) ^ 2 + (-0.55)')