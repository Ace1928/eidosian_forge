import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_param_mut(self):
    tests = [(self.param_mut, self.invalid, NotImplemented), (self.param_mut, self.asbinary, LinearExpression([self.param_mut, self.minus_bin])), (self.param_mut, self.zero, self.param_mut), (self.param_mut, self.one, NPV_SumExpression([self.param_mut, -1])), (self.param_mut, self.native, NPV_SumExpression([self.param_mut, -5])), (self.param_mut, self.npv, NPV_SumExpression([self.param_mut, self.minus_npv])), (self.param_mut, self.param, NPV_SumExpression([self.param_mut, -6])), (self.param_mut, self.param_mut, NPV_SumExpression([self.param_mut, self.minus_param_mut])), (self.param_mut, self.var, LinearExpression([self.param_mut, self.minus_var])), (self.param_mut, self.mon_native, LinearExpression([self.param_mut, self.minus_mon_native])), (self.param_mut, self.mon_param, LinearExpression([self.param_mut, self.minus_mon_param])), (self.param_mut, self.mon_npv, LinearExpression([self.param_mut, self.minus_mon_npv])), (self.param_mut, self.linear, SumExpression([self.param_mut, self.minus_linear])), (self.param_mut, self.sum, SumExpression([self.param_mut, self.minus_sum])), (self.param_mut, self.other, SumExpression([self.param_mut, self.minus_other])), (self.param_mut, self.mutable_l0, self.param_mut), (self.param_mut, self.mutable_l1, LinearExpression([self.param_mut, self.minus_mon_npv])), (self.param_mut, self.mutable_l2, SumExpression([self.param_mut, self.minus_mutable_l2])), (self.param_mut, self.param0, self.param_mut), (self.param_mut, self.param1, NPV_SumExpression([self.param_mut, -1])), (self.param_mut, self.mutable_l3, NPV_SumExpression([self.param_mut, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)