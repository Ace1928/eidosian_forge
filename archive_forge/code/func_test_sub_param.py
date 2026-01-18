import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_param(self):
    tests = [(self.param, self.invalid, NotImplemented), (self.param, self.asbinary, LinearExpression([6, self.minus_bin])), (self.param, self.zero, 6), (self.param, self.one, 5), (self.param, self.native, 1), (self.param, self.npv, NPV_SumExpression([6, self.minus_npv])), (self.param, self.param, 0), (self.param, self.param_mut, NPV_SumExpression([6, self.minus_param_mut])), (self.param, self.var, LinearExpression([6, self.minus_var])), (self.param, self.mon_native, LinearExpression([6, self.minus_mon_native])), (self.param, self.mon_param, LinearExpression([6, self.minus_mon_param])), (self.param, self.mon_npv, LinearExpression([6, self.minus_mon_npv])), (self.param, self.linear, SumExpression([6, self.minus_linear])), (self.param, self.sum, SumExpression([6, self.minus_sum])), (self.param, self.other, SumExpression([6, self.minus_other])), (self.param, self.mutable_l0, 6), (self.param, self.mutable_l1, LinearExpression([6, self.minus_mon_npv])), (self.param, self.mutable_l2, SumExpression([6, self.minus_mutable_l2])), (self.param, self.param0, 6), (self.param, self.param1, 5), (self.param, self.mutable_l3, NPV_SumExpression([6, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)