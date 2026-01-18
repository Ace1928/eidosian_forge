import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_linear(self):
    tests = [(self.linear, self.invalid, NotImplemented), (self.linear, self.asbinary, LinearExpression(self.linear.args + [self.minus_bin])), (self.linear, self.zero, self.linear), (self.linear, self.one, LinearExpression(self.linear.args + [-1])), (self.linear, self.native, LinearExpression(self.linear.args + [-5])), (self.linear, self.npv, LinearExpression(self.linear.args + [self.minus_npv])), (self.linear, self.param, LinearExpression(self.linear.args + [-6])), (self.linear, self.param_mut, LinearExpression(self.linear.args + [self.minus_param_mut])), (self.linear, self.var, LinearExpression(self.linear.args + [self.minus_var])), (self.linear, self.mon_native, LinearExpression(self.linear.args + [self.minus_mon_native])), (self.linear, self.mon_param, LinearExpression(self.linear.args + [self.minus_mon_param])), (self.linear, self.mon_npv, LinearExpression(self.linear.args + [self.minus_mon_npv])), (self.linear, self.linear, SumExpression([self.linear, self.minus_linear])), (self.linear, self.sum, SumExpression([self.linear, self.minus_sum])), (self.linear, self.other, SumExpression([self.linear, self.minus_other])), (self.linear, self.mutable_l0, self.linear), (self.linear, self.mutable_l1, LinearExpression(self.linear.args + [self.minus_mon_npv])), (self.linear, self.mutable_l2, SumExpression([self.linear, self.minus_mutable_l2])), (self.linear, self.param0, self.linear), (self.linear, self.param1, LinearExpression(self.linear.args + [-1])), (self.linear, self.mutable_l3, LinearExpression(self.linear.args + [self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)