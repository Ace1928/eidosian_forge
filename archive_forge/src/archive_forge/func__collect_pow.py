import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
def _collect_pow(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[1].__class__ in native_numeric_types:
        exponent = exp._args_[1]
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            exponent = value(exp._args_[1])
        else:
            exponent = exp._args_[1]
    else:
        res = _collect_standard_repn(exp._args_[1], 1, idMap, compute_values, verbose, quadratic)
        if not (res.nonl.__class__ in native_numeric_types and res.nonl == 0) or len(res.linear) > 0 or (quadratic and len(res.quadratic) > 0):
            return Results(nonl=multiplier * exp)
        exponent = res.constant
    if exponent.__class__ in native_numeric_types:
        if exponent == 0:
            return Results(constant=multiplier)
        elif exponent == 1:
            return _collect_standard_repn(exp._args_[0], multiplier, idMap, compute_values, verbose, quadratic)
        elif exponent == 2 and quadratic:
            res = _collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
            if not (res.nonl.__class__ in native_numeric_types and res.nonl == 0) or len(res.quadratic) > 0:
                return Results(nonl=multiplier * exp)
            elif compute_values and len(res.linear) == 0:
                return Results(constant=multiplier * res.constant ** exponent)
            else:
                ans = Results()
                has_constant = res.constant.__class__ not in native_numeric_types or res.constant != 0
                if has_constant:
                    ans.constant = multiplier * res.constant * res.constant
                keys = sorted(res.linear.keys(), reverse=True)
                while len(keys) > 0:
                    key1 = keys.pop()
                    coef1 = res.linear[key1]
                    if has_constant:
                        ans.linear[key1] = 2 * multiplier * coef1 * res.constant
                    ans.quadratic[key1, key1] = multiplier * coef1 * coef1
                    for key2 in keys:
                        coef2 = res.linear[key2]
                        ans.quadratic[key1, key2] = 2 * multiplier * coef1 * coef2
                return ans
    if exp._args_[0].__class__ in native_numeric_types or exp._args_[0].is_fixed():
        if compute_values:
            return Results(constant=multiplier * value(exp._args_[0]) ** exponent)
        else:
            return Results(constant=multiplier * exp)
    return Results(nonl=multiplier * exp)