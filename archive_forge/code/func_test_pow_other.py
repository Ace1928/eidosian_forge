import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_other(self):
    tests = [(self.other, self.invalid, NotImplemented), (self.other, self.asbinary, PowExpression((self.other, self.bin))), (self.other, self.zero, 1), (self.other, self.one, self.other), (self.other, self.native, PowExpression((self.other, 5))), (self.other, self.npv, PowExpression((self.other, self.npv))), (self.other, self.param, PowExpression((self.other, 6))), (self.other, self.param_mut, PowExpression((self.other, self.param_mut))), (self.other, self.var, PowExpression((self.other, self.var))), (self.other, self.mon_native, PowExpression((self.other, self.mon_native))), (self.other, self.mon_param, PowExpression((self.other, self.mon_param))), (self.other, self.mon_npv, PowExpression((self.other, self.mon_npv))), (self.other, self.linear, PowExpression((self.other, self.linear))), (self.other, self.sum, PowExpression((self.other, self.sum))), (self.other, self.other, PowExpression((self.other, self.other))), (self.other, self.mutable_l0, 1), (self.other, self.mutable_l1, PowExpression((self.other, self.mon_npv))), (self.other, self.mutable_l2, PowExpression((self.other, self.mutable_l2))), (self.other, self.param0, 1), (self.other, self.param1, self.other), (self.other, self.mutable_l3, PowExpression((self.other, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)