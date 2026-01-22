import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def add_enforced(self, ct: Union[ConstraintT, pd.Series], var: Union[Variable, pd.Series], value: Union[bool, pd.Series], name: Optional[str]=None) -> Union[EnforcedLinearConstraint, pd.Series]:
    """Adds a `ivar == ivalue => BoundedLinearExpression` to the model.

        Args:
          ct: A [`BoundedLinearExpression`](#boundedlinearexpression).
          var: The indicator variable
          value: the indicator value
          name: An optional name.

        Returns:
          An instance of the `Constraint` class.

        Note that a special treatment is done when the argument does not contain any
        variable, and thus evaluates to True or False.

        model.add_enforced(True, ivar, ivalue) will create a constraint 0 <= empty
        sum <= 0

        model.add_enforced(False, var, value) will create a constraint inf <=
        empty sum <= -inf

        you can check the if a constraint is always false (lb=inf, ub=-inf) by
        calling EnforcedLinearConstraint.is_always_false()
        """
    if isinstance(ct, _BoundedLinearExpr):
        return ct._add_enforced_linear_constraint(self.__helper, var, value, name)
    elif isinstance(ct, bool) and isinstance(var, Variable) and isinstance(value, bool):
        return _add_enforced_linear_constraint_to_helper(ct, self.__helper, var, value, name)
    elif isinstance(ct, pd.Series):
        ivar_series = _convert_to_var_series_and_validate_index(var, ct.index)
        ivalue_series = _convert_to_series_and_validate_index(value, ct.index)
        return pd.Series(index=ct.index, data=[_add_enforced_linear_constraint_to_helper(expr, self.__helper, ivar_series[i], ivalue_series[i], f'{name}[{i}]') for i, expr in zip(ct.index, ct)])
    else:
        raise TypeError('Not supported: Model.add_enforced(' + str(ct) + ')')