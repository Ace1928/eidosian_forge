import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_param0(self):
    tests = [(self.param0, self.invalid, NotImplemented), (self.param0, self.asbinary, self.minus_bin), (self.param0, self.zero, 0), (self.param0, self.one, -1), (self.param0, self.native, -5), (self.param0, self.npv, self.minus_npv), (self.param0, self.param, -6), (self.param0, self.param_mut, self.minus_param_mut), (self.param0, self.var, self.minus_var), (self.param0, self.mon_native, self.minus_mon_native), (self.param0, self.mon_param, self.minus_mon_param), (self.param0, self.mon_npv, self.minus_mon_npv), (self.param0, self.linear, self.minus_linear), (self.param0, self.sum, self.minus_sum), (self.param0, self.other, self.minus_other), (self.param0, self.mutable_l0, 0), (self.param0, self.mutable_l1, self.minus_mon_npv), (self.param0, self.mutable_l2, self.minus_mutable_l2), (self.param0, self.param0, 0), (self.param0, self.param1, -1), (self.param0, self.mutable_l3, self.minus_npv)]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)