import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_invalid(self):
    tests = [(self.invalid, self.invalid, NotImplemented), (self.invalid, self.asbinary, NotImplemented), (self.invalid, self.zero, NotImplemented), (self.invalid, self.one, NotImplemented), (self.invalid, self.native, NotImplemented), (self.invalid, self.npv, NotImplemented), (self.invalid, self.param, NotImplemented), (self.invalid, self.param_mut, NotImplemented), (self.invalid, self.var, NotImplemented), (self.invalid, self.mon_native, NotImplemented), (self.invalid, self.mon_param, NotImplemented), (self.invalid, self.mon_npv, NotImplemented), (self.invalid, self.linear, NotImplemented), (self.invalid, self.sum, NotImplemented), (self.invalid, self.other, NotImplemented), (self.invalid, self.mutable_l0, NotImplemented), (self.invalid, self.mutable_l1, NotImplemented), (self.invalid, self.mutable_l2, NotImplemented), (self.invalid, self.param0, NotImplemented), (self.invalid, self.param1, NotImplemented), (self.invalid, self.mutable_l3, NotImplemented)]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)