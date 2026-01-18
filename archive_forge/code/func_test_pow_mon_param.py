import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mon_param(self):
    tests = [(self.mon_param, self.invalid, NotImplemented), (self.mon_param, self.asbinary, PowExpression((self.mon_param, self.bin))), (self.mon_param, self.zero, 1), (self.mon_param, self.one, self.mon_param), (self.mon_param, self.native, PowExpression((self.mon_param, 5))), (self.mon_param, self.npv, PowExpression((self.mon_param, self.npv))), (self.mon_param, self.param, PowExpression((self.mon_param, 6))), (self.mon_param, self.param_mut, PowExpression((self.mon_param, self.param_mut))), (self.mon_param, self.var, PowExpression((self.mon_param, self.var))), (self.mon_param, self.mon_native, PowExpression((self.mon_param, self.mon_native))), (self.mon_param, self.mon_param, PowExpression((self.mon_param, self.mon_param))), (self.mon_param, self.mon_npv, PowExpression((self.mon_param, self.mon_npv))), (self.mon_param, self.linear, PowExpression((self.mon_param, self.linear))), (self.mon_param, self.sum, PowExpression((self.mon_param, self.sum))), (self.mon_param, self.other, PowExpression((self.mon_param, self.other))), (self.mon_param, self.mutable_l0, 1), (self.mon_param, self.mutable_l1, PowExpression((self.mon_param, self.mon_npv))), (self.mon_param, self.mutable_l2, PowExpression((self.mon_param, self.mutable_l2))), (self.mon_param, self.param0, 1), (self.mon_param, self.param1, self.mon_param), (self.mon_param, self.mutable_l3, PowExpression((self.mon_param, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)