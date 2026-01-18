import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_linear(self):
    tests = [(self.linear, self.invalid, NotImplemented), (self.linear, self.asbinary, PowExpression((self.linear, self.bin))), (self.linear, self.zero, 1), (self.linear, self.one, self.linear), (self.linear, self.native, PowExpression((self.linear, 5))), (self.linear, self.npv, PowExpression((self.linear, self.npv))), (self.linear, self.param, PowExpression((self.linear, 6))), (self.linear, self.param_mut, PowExpression((self.linear, self.param_mut))), (self.linear, self.var, PowExpression((self.linear, self.var))), (self.linear, self.mon_native, PowExpression((self.linear, self.mon_native))), (self.linear, self.mon_param, PowExpression((self.linear, self.mon_param))), (self.linear, self.mon_npv, PowExpression((self.linear, self.mon_npv))), (self.linear, self.linear, PowExpression((self.linear, self.linear))), (self.linear, self.sum, PowExpression((self.linear, self.sum))), (self.linear, self.other, PowExpression((self.linear, self.other))), (self.linear, self.mutable_l0, 1), (self.linear, self.mutable_l1, PowExpression((self.linear, self.mon_npv))), (self.linear, self.mutable_l2, PowExpression((self.linear, self.mutable_l2))), (self.linear, self.param0, 1), (self.linear, self.param1, self.linear), (self.linear, self.mutable_l3, PowExpression((self.linear, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)