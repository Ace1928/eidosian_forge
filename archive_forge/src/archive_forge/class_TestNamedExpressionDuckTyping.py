import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
class TestNamedExpressionDuckTyping(unittest.TestCase):

    def check_api(self, obj):
        self.assertTrue(hasattr(obj, 'nargs'))
        self.assertTrue(hasattr(obj, 'arg'))
        self.assertTrue(hasattr(obj, 'args'))
        self.assertTrue(hasattr(obj, '__call__'))
        self.assertTrue(hasattr(obj, 'to_string'))
        self.assertTrue(hasattr(obj, 'PRECEDENCE'))
        self.assertTrue(hasattr(obj, '_to_string'))
        self.assertTrue(hasattr(obj, 'clone'))
        self.assertTrue(hasattr(obj, 'create_node_with_local_data'))
        self.assertTrue(hasattr(obj, 'is_constant'))
        self.assertTrue(hasattr(obj, 'is_fixed'))
        self.assertTrue(hasattr(obj, '_is_fixed'))
        self.assertTrue(hasattr(obj, 'is_potentially_variable'))
        self.assertTrue(hasattr(obj, 'is_named_expression_type'))
        self.assertTrue(hasattr(obj, 'is_expression_type'))
        self.assertTrue(hasattr(obj, 'polynomial_degree'))
        self.assertTrue(hasattr(obj, '_compute_polynomial_degree'))
        self.assertTrue(hasattr(obj, '_apply_operation'))

    def test_Objective(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Objective(expr=M.x)
        self.check_api(M.e)

    def test_Expression(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Expression(expr=M.x)
        self.check_api(M.e)

    def test_ExpressionIndex(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Expression([0])
        M.e[0] = M.x
        self.check_api(M.e[0])

    def test_expression(self):
        x = variable()
        e = expression()
        e.expr = x
        self.check_api(e)

    def test_objective(self):
        x = variable()
        e = objective()
        e.expr = x
        self.check_api(e)