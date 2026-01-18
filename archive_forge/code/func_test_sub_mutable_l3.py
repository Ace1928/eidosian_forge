import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_mutable_l3(self):
    tests = [(self.mutable_l3, self.invalid, NotImplemented), (self.mutable_l3, self.asbinary, LinearExpression(self.mutable_l3.args + [self.minus_bin])), (self.mutable_l3, self.zero, self.npv), (self.mutable_l3, self.one, NPV_SumExpression(self.mutable_l3.args + [-1])), (self.mutable_l3, self.native, NPV_SumExpression(self.mutable_l3.args + [-5])), (self.mutable_l3, self.npv, NPV_SumExpression(self.mutable_l3.args + [self.minus_npv])), (self.mutable_l3, self.param, NPV_SumExpression(self.mutable_l3.args + [-6])), (self.mutable_l3, self.param_mut, NPV_SumExpression(self.mutable_l3.args + [self.minus_param_mut])), (self.mutable_l3, self.var, LinearExpression(self.mutable_l3.args + [self.minus_var])), (self.mutable_l3, self.mon_native, LinearExpression(self.mutable_l3.args + [self.minus_mon_native])), (self.mutable_l3, self.mon_param, LinearExpression(self.mutable_l3.args + [self.minus_mon_param])), (self.mutable_l3, self.mon_npv, LinearExpression(self.mutable_l3.args + [self.minus_mon_npv])), (self.mutable_l3, self.linear, SumExpression(self.mutable_l3.args + [self.minus_linear])), (self.mutable_l3, self.sum, SumExpression(self.mutable_l3.args + [self.minus_sum])), (self.mutable_l3, self.other, SumExpression(self.mutable_l3.args + [self.minus_other])), (self.mutable_l3, self.mutable_l0, self.npv), (self.mutable_l3, self.mutable_l1, LinearExpression(self.mutable_l3.args + [self.minus_mon_npv])), (self.mutable_l3, self.mutable_l2, SumExpression(self.mutable_l3.args + [self.minus_mutable_l2])), (self.mutable_l3, self.param0, self.npv), (self.mutable_l3, self.param1, NPV_SumExpression(self.mutable_l3.args + [-1])), (self.mutable_l3, self.mutable_l3, NPV_SumExpression([NPV_SumExpression(self.mutable_l3.args), self.minus_npv]))]
    self._run_cases(tests, operator.sub)