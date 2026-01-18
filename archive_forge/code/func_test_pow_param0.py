import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_param0(self):
    tests = [(self.param0, self.invalid, NotImplemented), (self.param0, self.asbinary, PowExpression((0, self.bin))), (self.param0, self.zero, 1), (self.param0, self.one, 0), (self.param0, self.native, 0), (self.param0, self.npv, NPV_PowExpression((0, self.npv))), (self.param0, self.param, 0), (self.param0, self.param_mut, NPV_PowExpression((0, self.param_mut))), (self.param0, self.var, PowExpression((0, self.var))), (self.param0, self.mon_native, PowExpression((0, self.mon_native))), (self.param0, self.mon_param, PowExpression((0, self.mon_param))), (self.param0, self.mon_npv, PowExpression((0, self.mon_npv))), (self.param0, self.linear, PowExpression((0, self.linear))), (self.param0, self.sum, PowExpression((0, self.sum))), (self.param0, self.other, PowExpression((0, self.other))), (self.param0, self.mutable_l0, 1), (self.param0, self.mutable_l1, PowExpression((0, self.mon_npv))), (self.param0, self.mutable_l2, PowExpression((0, self.mutable_l2))), (self.param0, self.param0, 1), (self.param0, self.param1, 0), (self.param0, self.mutable_l3, NPV_PowExpression((0, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)