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
class WalkerTests(unittest.TestCase):

    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(M.w[1]) + M.w[1] * M.w[2] + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        e = M.x
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(self, sin(M.w[1]) + M.w[1] * M.w[2] + 3 <= 0, f)

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()
        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(self, inequality(0, sin(M.w[1]) + M.w[1] * M.w[2] + 3, 1), f)

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)
        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
        assertExpressionsEqual(self, LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.w.values()]), f)
        del M.w
        M.w = VarList()
        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, 2 * LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
        assertExpressionsEqual(self, 2 * LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.w.values()]), f)

    def test_replacement_linear_expression_with_constant(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression(linear_coefs=[2], linear_vars=[m.x])
        e += m.y
        sub_map = dict()
        sub_map[id(m.x)] = 5
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(self, e2, LinearExpression([10, MonomialTermExpression((1, m.y))]))
        e = LinearExpression(linear_coefs=[2, 3], linear_vars=[m.x, m.y])
        sub_map = dict()
        sub_map[id(m.x)] = 5
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(self, e2, LinearExpression([10, MonomialTermExpression((3, m.y))]))

    def test_replacement_linear_expression_with_nonlinear(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression(linear_coefs=[2, 3], linear_vars=[m.x, m.y])
        sub_map = dict()
        sub_map[id(m.x)] = m.x ** 2
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(self, e2, SumExpression([2 * m.x ** 2, 3 * m.y]))

    def test_replace_expressions_with_monomial_term(self):
        M = ConcreteModel()
        M.x = Var()
        e = 2.0 * M.x
        substitution_map = {id(M.x): 3.0 * M.x}
        new_e = replace_expressions(e, substitution_map=substitution_map)
        self.assertEqual('6.0*x', str(new_e))

    def test_identify_components(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = Var()
        e = sin(M.x) + M.x * M.w + 3
        v = list((str(v) for v in identify_components(e, set([M.x.__class__]))))
        self.assertEqual(v, ['x', 'w'])
        v = list((str(v) for v in identify_components(e, [M.x.__class__])))
        self.assertEqual(v, ['x', 'w'])

    def test_identify_variables(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = Var()
        M.w = 2
        M.w.fixed = True
        e = sin(M.x) + M.x * M.w + 3
        v = list((str(v) for v in identify_variables(e)))
        self.assertEqual(v, ['x', 'w'])
        v = list((str(v) for v in identify_variables(e, include_fixed=False)))
        self.assertEqual(v, ['x'])

    def test_expression_to_string(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = Var()
        e = sin(M.x) + M.x * M.w + 3
        self.assertEqual('sin(x) + x*w + 3', expression_to_string(e))
        M.w = 2
        M.w.fixed = True
        self.assertEqual('sin(x) + x*2 + 3', expression_to_string(e, compute_values=True))

    def test_expression_component_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=m.x * m.y)
        m.f = Expression(expr=m.e)
        e = m.x + m.f * m.y
        self.assertEqual('x + ((x*y))*y', str(e))
        self.assertEqual('x + ((x*y))*y', expression_to_string(e))