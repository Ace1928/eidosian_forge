import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class WalkerTests3(unittest.TestCase):

    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.w = VarList()
        e = M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3 <= 0, f)

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()
        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(self, inequality(0, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3, 1), f)

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        e = M.z * M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertIs(e.__class__, MonomialTermExpression)
        self.assertIs(f.__class__, ProductExpression)
        self.assertTrue(f.arg(0).is_potentially_variable())
        assertExpressionsEqual(self, M.z * M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1] * M.x, f)

    def test_replacement_walker6(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        e = M.z * 2 * 3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(not e.is_potentially_variable())
        self.assertTrue(f.is_potentially_variable())
        assertExpressionsEqual(self, M.z * 2 * 3, e)
        assertExpressionsEqual(self, ProductExpression([ProductExpression([2 * M.w[1], 2]), 3]), f)

    def test_replacement_walker7(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        M.e = Expression(expr=M.z * 2)
        e = M.x * M.e
        self.assertTrue(e.arg(1).is_potentially_variable())
        self.assertTrue(not e.arg(1).arg(0).is_potentially_variable())
        assertExpressionsEqual(self, ProductExpression([M.x, NPV_ProductExpression([M.z, 2])]), e, include_named_exprs=False)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(e.__class__ is ProductExpression)
        self.assertTrue(f.__class__ is ProductExpression)
        self.assertEqual(id(e), id(f))
        self.assertTrue(f.arg(1).is_potentially_variable())
        self.assertTrue(f.arg(1).arg(0).is_potentially_variable())
        assertExpressionsEqual(self, M.x * ProductExpression([2 * M.w[1], 2]), f, include_named_exprs=False)

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)
        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
        assertExpressionsEqual(self, 2 * M.w[1] * M.x[0] + 2 * M.w[2] * M.x[1] + 2 * M.w[3] * M.x[2], f)
        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, 2 * LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
        assertExpressionsEqual(self, 2 * (2 * M.w[4] * M.x[0] + 2 * M.w[5] * M.x[1] + 2 * M.w[6] * M.x[2]), f)