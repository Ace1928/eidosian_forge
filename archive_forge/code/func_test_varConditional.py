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