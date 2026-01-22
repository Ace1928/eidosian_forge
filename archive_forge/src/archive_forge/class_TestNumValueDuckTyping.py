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
class TestNumValueDuckTyping(unittest.TestCase):

    def check_api(self, obj):
        self.assertTrue(hasattr(obj, 'is_fixed'))
        self.assertTrue(hasattr(obj, 'is_constant'))
        self.assertTrue(hasattr(obj, 'is_parameter_type'))
        self.assertTrue(hasattr(obj, 'is_potentially_variable'))
        self.assertTrue(hasattr(obj, 'is_variable_type'))
        self.assertTrue(hasattr(obj, 'is_named_expression_type'))
        self.assertTrue(hasattr(obj, 'is_expression_type'))
        self.assertTrue(hasattr(obj, '_compute_polynomial_degree'))
        self.assertTrue(hasattr(obj, '__call__'))
        self.assertTrue(hasattr(obj, 'to_string'))

    def test_Param(self):
        M = ConcreteModel()
        M.x = Param()
        self.check_api(M.x)

    def test_MutableParam(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        self.check_api(M.x)

    def test_MutableParamIndex(self):
        M = ConcreteModel()
        M.x = Param([0], initialize=10, mutable=True)
        self.check_api(M.x[0])

    def test_Var(self):
        M = ConcreteModel()
        M.x = Var()
        self.check_api(M.x)

    def test_VarIndex(self):
        M = ConcreteModel()
        M.x = Var([0])
        self.check_api(M.x[0])

    def test_variable(self):
        x = variable()
        self.check_api(x)