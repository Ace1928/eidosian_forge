import logging
import math
import operator
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from pyomo.core.expr.visitor import clone_expression
from pyomo.environ import ConcreteModel, Param, Var, BooleanVar
from pyomo.gdp import Disjunct
def test_unreachable_dispatchers(self):

    class CustomAsNumeric(NumericValue):

        def __init__(self, val):
            self._value = val

        def is_numeric_type(self):
            return False

        def as_numeric(self):
            return self._value
    obj = CustomAsNumeric(self.var)
    e = abs(obj)
    assertExpressionsEqual(self, AbsExpression((self.var,)), e)
    e = obj ** 2
    assertExpressionsEqual(self, PowExpression((self.var, 2)), e)
    e = obj + obj
    assertExpressionsEqual(self, LinearExpression((self.mon_var, self.mon_var)), e)