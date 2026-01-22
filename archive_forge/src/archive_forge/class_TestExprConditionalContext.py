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
class TestExprConditionalContext(unittest.TestCase):

    def checkCondition(self, expr, expectedValue, use_value=False):
        if use_value:
            expr = value(expr)
        try:
            if expr:
                if expectedValue != True:
                    self.fail('__bool__ returned the wrong condition value (expected %s)' % expectedValue)
            elif expectedValue != False:
                self.fail('__bool__ returned the wrong condition value (expected %s)' % expectedValue)
            if expectedValue is None:
                self.fail('Expected ValueError because component was undefined')
        except (ValueError, PyomoException):
            if expectedValue is not None:
                raise

    def test_immutable_paramConditional(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(model.p > 0, True)
        instance = model.create_instance()
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(model.p > 0, True)
        instance = model.create_instance()
        self.checkCondition(instance.p > 0, True)
        self.checkCondition(instance.p > 2, False)
        self.checkCondition(instance.p >= 1, True)
        self.checkCondition(instance.p >= 2, False)
        self.checkCondition(instance.p < 2, True)
        self.checkCondition(instance.p < 0, False)
        self.checkCondition(instance.p <= 1, True)
        self.checkCondition(instance.p <= 0, False)
        self.checkCondition(instance.p == 1, True)
        self.checkCondition(instance.p == 2, False)

    def test_immutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(0 < model.p, True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <=  p\\) to bool.'):
            self.checkCondition(0 <= model.p, True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(p  <  1\\) to bool.'):
            self.checkCondition(1 > model.p, True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(p  <=  1\\) to bool.'):
            self.checkCondition(1 >= model.p, True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  ==  p\\) to bool.'):
            self.checkCondition(0 == model.p, None)
        self.checkCondition(0 < model.p, True, use_value=True)
        self.checkCondition(0 <= model.p, True, use_value=True)
        self.checkCondition(1 > model.p, True, use_value=True)
        self.checkCondition(1 >= model.p, True, use_value=True)
        self.checkCondition(0 == model.p, None, use_value=True)
        instance = model.create_instance()
        self.checkCondition(0 < instance.p, True)
        self.checkCondition(2 < instance.p, False)
        self.checkCondition(1 <= instance.p, True)
        self.checkCondition(instance.p > 0, True)
        self.checkCondition(instance.p > 2, False)
        self.checkCondition(instance.p >= 1, True)
        self.checkCondition(instance.p >= 2, False)
        self.checkCondition(instance.p < 2, True)
        self.checkCondition(instance.p < 0, False)
        self.checkCondition(instance.p <= 1, True)
        self.checkCondition(instance.p <= 0, False)
        self.checkCondition(instance.p == 1, True)
        self.checkCondition(instance.p == 2, False)

    def test_immutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(0 < model.p, True)
        instance = model.create_instance()
        self.checkCondition(0 < instance.p, True)
        self.checkCondition(2 < instance.p, False)
        self.checkCondition(1 <= instance.p, True)
        self.checkCondition(2 <= instance.p, False)
        self.checkCondition(2 > instance.p, True)
        self.checkCondition(0 > instance.p, False)
        self.checkCondition(1 >= instance.p, True)
        self.checkCondition(0 >= instance.p, False)
        self.checkCondition(1 == instance.p, True)
        self.checkCondition(2 == instance.p, False)

    def test_mutable_paramConditional(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(model.p > 0, True)
        instance = model.create_instance()
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p > 0, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p > 2, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p >= 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p >= 2, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p < 2, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p < 0, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p <= 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p <= 0, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p == 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.p == 2, False)
        self.checkCondition(instance.p > 0, True, use_value=True)
        self.checkCondition(instance.p > 2, False, use_value=True)
        self.checkCondition(instance.p >= 1, True, use_value=True)
        self.checkCondition(instance.p >= 2, False, use_value=True)
        self.checkCondition(instance.p < 2, True, use_value=True)
        self.checkCondition(instance.p < 0, False, use_value=True)
        self.checkCondition(instance.p <= 1, True, use_value=True)
        self.checkCondition(instance.p <= 0, False, use_value=True)
        self.checkCondition(instance.p == 1, True, use_value=True)
        self.checkCondition(instance.p == 2, False, use_value=True)

    def test_mutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=True)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  p\\) to bool.'):
            self.checkCondition(0 < model.p, True)
        instance = model.create_instance()
        with self.assertRaises(PyomoException):
            self.checkCondition(0 < instance.p, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 < instance.p, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 <= instance.p, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 <= instance.p, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 > instance.p, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(0 > instance.p, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 >= instance.p, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(0 >= instance.p, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 == instance.p, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 == instance.p, False)
        self.checkCondition(0 < instance.p, True, use_value=True)
        self.checkCondition(2 < instance.p, False, use_value=True)
        self.checkCondition(1 <= instance.p, True, use_value=True)
        self.checkCondition(2 <= instance.p, False, use_value=True)
        self.checkCondition(2 > instance.p, True, use_value=True)
        self.checkCondition(0 > instance.p, False, use_value=True)
        self.checkCondition(1 >= instance.p, True, use_value=True)
        self.checkCondition(0 >= instance.p, False, use_value=True)
        self.checkCondition(1 == instance.p, True, use_value=True)
        self.checkCondition(2 == instance.p, False, use_value=True)

    def test_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  v\\) to bool.'):
            self.checkCondition(model.v > 0, True)
        instance = model.create_instance()
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v > 0, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v > 2, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v >= 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v >= 2, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v < 2, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v < 0, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v <= 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v <= 0, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v == 1, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(instance.v == 2, False)
        self.checkCondition(instance.v > 0, True, use_value=True)
        self.checkCondition(instance.v > 2, False, use_value=True)
        self.checkCondition(instance.v >= 1, True, use_value=True)
        self.checkCondition(instance.v >= 2, False, use_value=True)
        self.checkCondition(instance.v < 2, True, use_value=True)
        self.checkCondition(instance.v < 0, False, use_value=True)
        self.checkCondition(instance.v <= 1, True, use_value=True)
        self.checkCondition(instance.v <= 0, False, use_value=True)
        self.checkCondition(instance.v == 1, True, use_value=True)
        self.checkCondition(instance.v == 2, False, use_value=True)

    def test_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo expression \\(0  <  v\\) to bool.'):
            self.checkCondition(0 < model.v, True)
        instance = model.create_instance()
        with self.assertRaises(PyomoException):
            self.checkCondition(0 < instance.v, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 < instance.v, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 <= instance.v, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 <= instance.v, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 > instance.v, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(0 > instance.v, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 >= instance.v, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(0 >= instance.v, False)
        with self.assertRaises(PyomoException):
            self.checkCondition(1 == instance.v, True)
        with self.assertRaises(PyomoException):
            self.checkCondition(2 == instance.v, False)
        self.checkCondition(0 < instance.v, True, use_value=True)
        self.checkCondition(2 < instance.v, False, use_value=True)
        self.checkCondition(1 <= instance.v, True, use_value=True)
        self.checkCondition(2 <= instance.v, False, use_value=True)
        self.checkCondition(2 > instance.v, True, use_value=True)
        self.checkCondition(0 > instance.v, False, use_value=True)
        self.checkCondition(1 >= instance.v, True, use_value=True)
        self.checkCondition(0 >= instance.v, False, use_value=True)
        self.checkCondition(1 == instance.v, True, use_value=True)
        self.checkCondition(2 == instance.v, False, use_value=True)

    def test_eval_sub_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v) > 0, None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v) >= 0, None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v) < 1, None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v) <= 1, None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v) == 0, None)
        instance = model.create_instance()
        self.checkCondition(value(instance.v) > 0, True)
        self.checkCondition(value(instance.v) > 2, False)
        self.checkCondition(value(instance.v) >= 1, True)
        self.checkCondition(value(instance.v) >= 2, False)
        self.checkCondition(value(instance.v) < 2, True)
        self.checkCondition(value(instance.v) < 0, False)
        self.checkCondition(value(instance.v) <= 1, True)
        self.checkCondition(value(instance.v) <= 0, False)
        self.checkCondition(value(instance.v) == 1, True)
        self.checkCondition(value(instance.v) == 2, False)

    def test_eval_sub_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(0 < value(model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(0 <= value(model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(1 > value(model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(1 >= value(model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(0 == value(model.v), None)
        instance = model.create_instance()
        self.checkCondition(0 < value(instance.v), True)
        self.checkCondition(2 < value(instance.v), False)
        self.checkCondition(1 <= value(instance.v), True)
        self.checkCondition(2 <= value(instance.v), False)
        self.checkCondition(2 > value(instance.v), True)
        self.checkCondition(0 > value(instance.v), False)
        self.checkCondition(1 >= value(instance.v), True)
        self.checkCondition(0 >= value(instance.v), False)
        self.checkCondition(1 == value(instance.v), True)
        self.checkCondition(2 == value(instance.v), False)

    def test_eval_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v > 0), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v >= 0), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(model.v == 0), None)
        instance = model.create_instance()
        self.checkCondition(value(instance.v > 0), True)
        self.checkCondition(value(instance.v > 2), False)
        self.checkCondition(value(instance.v >= 1), True)
        self.checkCondition(value(instance.v >= 2), False)
        self.checkCondition(value(instance.v == 1), True)
        self.checkCondition(value(instance.v == 2), False)

    def test_eval_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(0 < model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(0 <= model.v), None)
        with self.assertRaisesRegex(RuntimeError, "Cannot access property 'value' on AbstractScalarVar 'v' before it has been constructed"):
            self.checkCondition(value(0 == model.v), None)
        instance = model.create_instance()
        self.checkCondition(value(0 < instance.v), True)
        self.checkCondition(value(2 < instance.v), False)
        self.checkCondition(value(1 <= instance.v), True)
        self.checkCondition(value(2 <= instance.v), False)
        self.checkCondition(value(1 == instance.v), True)
        self.checkCondition(value(2 == instance.v), False)