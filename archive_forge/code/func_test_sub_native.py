import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_native(self):
    tests = [(self.native, self.invalid, NotImplemented), (self.native, self.asbinary, LinearExpression([5, self.minus_bin])), (self.native, self.zero, 5), (self.native, self.one, 4), (self.native, self.native, 0), (self.native, self.npv, NPV_SumExpression([5, self.minus_npv])), (self.native, self.param, -1), (self.native, self.param_mut, NPV_SumExpression([5, self.minus_param_mut])), (self.native, self.var, LinearExpression([5, self.minus_var])), (self.native, self.mon_native, LinearExpression([5, self.minus_mon_native])), (self.native, self.mon_param, LinearExpression([5, self.minus_mon_param])), (self.native, self.mon_npv, LinearExpression([5, self.minus_mon_npv])), (self.native, self.linear, SumExpression([5, self.minus_linear])), (self.native, self.sum, SumExpression([5, self.minus_sum])), (self.native, self.other, SumExpression([5, self.minus_other])), (self.native, self.mutable_l0, 5), (self.native, self.mutable_l1, LinearExpression([5, self.minus_mon_npv])), (self.native, self.mutable_l2, SumExpression([5, self.minus_mutable_l2])), (self.native, self.param0, 5), (self.native, self.param1, 4), (self.native, self.mutable_l3, NPV_SumExpression([5, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)