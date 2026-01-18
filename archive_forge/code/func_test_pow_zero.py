import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_zero(self):
    tests = [(self.zero, self.invalid, NotImplemented), (self.zero, self.asbinary, PowExpression((0, self.bin))), (self.zero, self.zero, 1), (self.zero, self.one, 0), (self.zero, self.native, 0), (self.zero, self.npv, NPV_PowExpression((0, self.npv))), (self.zero, self.param, 0), (self.zero, self.param_mut, NPV_PowExpression((0, self.param_mut))), (self.zero, self.var, PowExpression((0, self.var))), (self.zero, self.mon_native, PowExpression((0, self.mon_native))), (self.zero, self.mon_param, PowExpression((0, self.mon_param))), (self.zero, self.mon_npv, PowExpression((0, self.mon_npv))), (self.zero, self.linear, PowExpression((0, self.linear))), (self.zero, self.sum, PowExpression((0, self.sum))), (self.zero, self.other, PowExpression((0, self.other))), (self.zero, self.mutable_l0, 1), (self.zero, self.mutable_l1, PowExpression((0, self.mon_npv))), (self.zero, self.mutable_l2, PowExpression((0, self.mutable_l2))), (self.zero, self.param0, 1), (self.zero, self.param1, 0), (self.zero, self.mutable_l3, NPV_PowExpression((0, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)