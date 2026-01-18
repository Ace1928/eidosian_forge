import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_var(self):
    tests = [(self.var, self.invalid, NotImplemented), (self.var, self.asbinary, PowExpression((self.var, self.bin))), (self.var, self.zero, 1), (self.var, self.one, self.var), (self.var, self.native, PowExpression((self.var, 5))), (self.var, self.npv, PowExpression((self.var, self.npv))), (self.var, self.param, PowExpression((self.var, 6))), (self.var, self.param_mut, PowExpression((self.var, self.param_mut))), (self.var, self.var, PowExpression((self.var, self.var))), (self.var, self.mon_native, PowExpression((self.var, self.mon_native))), (self.var, self.mon_param, PowExpression((self.var, self.mon_param))), (self.var, self.mon_npv, PowExpression((self.var, self.mon_npv))), (self.var, self.linear, PowExpression((self.var, self.linear))), (self.var, self.sum, PowExpression((self.var, self.sum))), (self.var, self.other, PowExpression((self.var, self.other))), (self.var, self.mutable_l0, 1), (self.var, self.mutable_l1, PowExpression((self.var, self.mon_npv))), (self.var, self.mutable_l2, PowExpression((self.var, self.mutable_l2))), (self.var, self.param0, 1), (self.var, self.param1, self.var), (self.var, self.mutable_l3, PowExpression((self.var, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)