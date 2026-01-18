import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_mon_native(self):
    tests = [(self.mon_native, self.invalid, NotImplemented), (self.mon_native, self.asbinary, LinearExpression([self.mon_native, self.minus_bin])), (self.mon_native, self.zero, self.mon_native), (self.mon_native, self.one, LinearExpression([self.mon_native, -1])), (self.mon_native, self.native, LinearExpression([self.mon_native, -5])), (self.mon_native, self.npv, LinearExpression([self.mon_native, self.minus_npv])), (self.mon_native, self.param, LinearExpression([self.mon_native, -6])), (self.mon_native, self.param_mut, LinearExpression([self.mon_native, self.minus_param_mut])), (self.mon_native, self.var, LinearExpression([self.mon_native, self.minus_var])), (self.mon_native, self.mon_native, LinearExpression([self.mon_native, self.minus_mon_native])), (self.mon_native, self.mon_param, LinearExpression([self.mon_native, self.minus_mon_param])), (self.mon_native, self.mon_npv, LinearExpression([self.mon_native, self.minus_mon_npv])), (self.mon_native, self.linear, SumExpression([self.mon_native, self.minus_linear])), (self.mon_native, self.sum, SumExpression([self.mon_native, self.minus_sum])), (self.mon_native, self.other, SumExpression([self.mon_native, self.minus_other])), (self.mon_native, self.mutable_l0, self.mon_native), (self.mon_native, self.mutable_l1, LinearExpression([self.mon_native, self.minus_mon_npv])), (self.mon_native, self.mutable_l2, SumExpression([self.mon_native, self.minus_mutable_l2])), (self.mon_native, self.param0, self.mon_native), (self.mon_native, self.param1, LinearExpression([self.mon_native, -1])), (self.mon_native, self.mutable_l3, LinearExpression([self.mon_native, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)