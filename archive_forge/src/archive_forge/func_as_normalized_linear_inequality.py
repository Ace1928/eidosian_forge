import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def as_normalized_linear_inequality(bounded_expr: Optional[Union[bool, BoundedLinearTypes]]=None, *, lb: Optional[float]=None, ub: Optional[float]=None, expr: Optional[LinearTypes]=None) -> NormalizedLinearInequality:
    """Converts a linear constraint to a NormalizedLinearInequality.

    The simplest way to specify the constraint is by passing a one-sided or
    two-sided linear inequality as in:
      * as_normalized_linear_inequality(x + y + 1.0 <= 2.0),
      * as_normalized_linear_inequality(x + y >= 2.0), or
      * as_normalized_linear_inequality((1.0 <= x + y) <= 2.0).

    Note the extra parenthesis for two-sided linear inequalities, which is
    required due to some language limitations (see
    https://peps.python.org/pep-0335/ and https://peps.python.org/pep-0535/).
    If the parenthesis are omitted, a TypeError will be raised explaining the
    issue (if this error was not raised the first inequality would have been
    silently ignored because of the noted language limitations).

    The second way to specify the constraint is by setting lb, ub, and/o expr as
    in:
      * as_normalized_linear_inequality(expr=x + y + 1.0, ub=2.0),
      * as_normalized_linear_inequality(expr=x + y, lb=2.0),
      * as_normalized_linear_inequality(expr=x + y, lb=1.0, ub=2.0), or
      * as_normalized_linear_inequality(lb=1.0).
    Omitting lb is equivalent to setting it to -math.inf and omiting ub is
    equivalent to setting it to math.inf.

    These two alternatives are exclusive and a combined call like:
      * as_normalized_linear_inequality(x + y <= 2.0, lb=1.0), or
      * as_normalized_linear_inequality(x + y <= 2.0, ub=math.inf)
    will raise a ValueError. A ValueError is also raised if expr's offset is
    infinite.

    Args:
      bounded_expr: a linear inequality describing the constraint. Cannot be
        specified together with lb, ub, or expr.
      lb: The constraint's lower bound if bounded_expr is omitted (if both
        bounder_expr and lb are omitted, the lower bound is -math.inf).
      ub: The constraint's upper bound if bounded_expr is omitted (if both
        bounder_expr and ub are omitted, the upper bound is math.inf).
      expr: The constraint's linear expression if bounded_expr is omitted.

    Returns:
      A NormalizedLinearInequality representing the linear constraint.
    """
    if bounded_expr is None:
        if lb is None:
            lb = -math.inf
        if ub is None:
            ub = math.inf
        if expr is None:
            return NormalizedLinearInequality(lb=lb, ub=ub, expr=0)
        if not isinstance(expr, (LinearBase, int, float)):
            raise TypeError(f'unsupported type for expr argument: {type(expr).__name__!r}')
        return NormalizedLinearInequality(lb=lb, ub=ub, expr=expr)
    if isinstance(bounded_expr, bool):
        raise TypeError('unsupported type for bounded_expr argument: bool. This error can occur when trying to add != constraints (which are not supported) or inequalities/equalities with constant left-hand-side and right-hand-side (which are redundant or make a model infeasible).')
    if not isinstance(bounded_expr, (LowerBoundedLinearExpression, UpperBoundedLinearExpression, BoundedLinearExpression, VarEqVar)):
        raise TypeError(f'unsupported type for bounded_expr: {type(bounded_expr).__name__!r}')
    if lb is not None:
        raise ValueError('lb cannot be specified together with a linear inequality')
    if ub is not None:
        raise ValueError('ub cannot be specified together with a linear inequality')
    if expr is not None:
        raise ValueError('expr cannot be specified together with a linear inequality')
    if isinstance(bounded_expr, VarEqVar):
        return NormalizedLinearInequality(lb=0.0, ub=0.0, expr=bounded_expr.first_variable - bounded_expr.second_variable)
    if isinstance(bounded_expr, (LowerBoundedLinearExpression, BoundedLinearExpression)):
        lb = bounded_expr.lower_bound
    else:
        lb = -math.inf
    if isinstance(bounded_expr, (UpperBoundedLinearExpression, BoundedLinearExpression)):
        ub = bounded_expr.upper_bound
    else:
        ub = math.inf
    return NormalizedLinearInequality(lb=lb, ub=ub, expr=bounded_expr.expression)