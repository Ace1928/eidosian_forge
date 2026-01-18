import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_param1(self):
    tests = [(self.param1, self.invalid, NotImplemented), (self.param1, self.asbinary, PowExpression((1, self.bin))), (self.param1, self.zero, 1), (self.param1, self.one, 1), (self.param1, self.native, 1), (self.param1, self.npv, NPV_PowExpression((1, self.npv))), (self.param1, self.param, 1), (self.param1, self.param_mut, NPV_PowExpression((1, self.param_mut))), (self.param1, self.var, PowExpression((1, self.var))), (self.param1, self.mon_native, PowExpression((1, self.mon_native))), (self.param1, self.mon_param, PowExpression((1, self.mon_param))), (self.param1, self.mon_npv, PowExpression((1, self.mon_npv))), (self.param1, self.linear, PowExpression((1, self.linear))), (self.param1, self.sum, PowExpression((1, self.sum))), (self.param1, self.other, PowExpression((1, self.other))), (self.param1, self.mutable_l0, 1), (self.param1, self.mutable_l1, PowExpression((1, self.mon_npv))), (self.param1, self.mutable_l2, PowExpression((1, self.mutable_l2))), (self.param1, self.param0, 1), (self.param1, self.param1, 1), (self.param1, self.mutable_l3, NPV_PowExpression((1, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)