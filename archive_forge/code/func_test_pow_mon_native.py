import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mon_native(self):
    tests = [(self.mon_native, self.invalid, NotImplemented), (self.mon_native, self.asbinary, PowExpression((self.mon_native, self.bin))), (self.mon_native, self.zero, 1), (self.mon_native, self.one, self.mon_native), (self.mon_native, self.native, PowExpression((self.mon_native, 5))), (self.mon_native, self.npv, PowExpression((self.mon_native, self.npv))), (self.mon_native, self.param, PowExpression((self.mon_native, 6))), (self.mon_native, self.param_mut, PowExpression((self.mon_native, self.param_mut))), (self.mon_native, self.var, PowExpression((self.mon_native, self.var))), (self.mon_native, self.mon_native, PowExpression((self.mon_native, self.mon_native))), (self.mon_native, self.mon_param, PowExpression((self.mon_native, self.mon_param))), (self.mon_native, self.mon_npv, PowExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.linear, PowExpression((self.mon_native, self.linear))), (self.mon_native, self.sum, PowExpression((self.mon_native, self.sum))), (self.mon_native, self.other, PowExpression((self.mon_native, self.other))), (self.mon_native, self.mutable_l0, 1), (self.mon_native, self.mutable_l1, PowExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.mutable_l2, PowExpression((self.mon_native, self.mutable_l2))), (self.mon_native, self.param0, 1), (self.mon_native, self.param1, self.mon_native), (self.mon_native, self.mutable_l3, PowExpression((self.mon_native, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)