import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_param(self):
    tests = [(self.param, self.invalid, NotImplemented), (self.param, self.asbinary, PowExpression((6, self.bin))), (self.param, self.zero, 1), (self.param, self.one, 6), (self.param, self.native, 7776), (self.param, self.npv, NPV_PowExpression((6, self.npv))), (self.param, self.param, 46656), (self.param, self.param_mut, NPV_PowExpression((6, self.param_mut))), (self.param, self.var, PowExpression((6, self.var))), (self.param, self.mon_native, PowExpression((6, self.mon_native))), (self.param, self.mon_param, PowExpression((6, self.mon_param))), (self.param, self.mon_npv, PowExpression((6, self.mon_npv))), (self.param, self.linear, PowExpression((6, self.linear))), (self.param, self.sum, PowExpression((6, self.sum))), (self.param, self.other, PowExpression((6, self.other))), (self.param, self.mutable_l0, 1), (self.param, self.mutable_l1, PowExpression((6, self.mon_npv))), (self.param, self.mutable_l2, PowExpression((6, self.mutable_l2))), (self.param, self.param0, 1), (self.param, self.param1, 6), (self.param, self.mutable_l3, NPV_PowExpression((6, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)