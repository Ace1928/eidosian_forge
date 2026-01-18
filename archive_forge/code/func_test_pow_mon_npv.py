import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mon_npv(self):
    tests = [(self.mon_npv, self.invalid, NotImplemented), (self.mon_npv, self.asbinary, PowExpression((self.mon_npv, self.bin))), (self.mon_npv, self.zero, 1), (self.mon_npv, self.one, self.mon_npv), (self.mon_npv, self.native, PowExpression((self.mon_npv, 5))), (self.mon_npv, self.npv, PowExpression((self.mon_npv, self.npv))), (self.mon_npv, self.param, PowExpression((self.mon_npv, 6))), (self.mon_npv, self.param_mut, PowExpression((self.mon_npv, self.param_mut))), (self.mon_npv, self.var, PowExpression((self.mon_npv, self.var))), (self.mon_npv, self.mon_native, PowExpression((self.mon_npv, self.mon_native))), (self.mon_npv, self.mon_param, PowExpression((self.mon_npv, self.mon_param))), (self.mon_npv, self.mon_npv, PowExpression((self.mon_npv, self.mon_npv))), (self.mon_npv, self.linear, PowExpression((self.mon_npv, self.linear))), (self.mon_npv, self.sum, PowExpression((self.mon_npv, self.sum))), (self.mon_npv, self.other, PowExpression((self.mon_npv, self.other))), (self.mon_npv, self.mutable_l0, 1), (self.mon_npv, self.mutable_l1, PowExpression((self.mon_npv, self.mon_npv))), (self.mon_npv, self.mutable_l2, PowExpression((self.mon_npv, self.mutable_l2))), (self.mon_npv, self.param0, 1), (self.mon_npv, self.param1, self.mon_npv), (self.mon_npv, self.mutable_l3, PowExpression((self.mon_npv, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)