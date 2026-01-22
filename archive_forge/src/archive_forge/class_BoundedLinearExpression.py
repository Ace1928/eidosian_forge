import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class BoundedLinearExpression:
    """Represents a linear constraint: `lb <= linear expression <= ub`.

    The only use of this class is to be added to the CpModel through
    `CpModel.add(expression)`, as in:

        model.add(x + 2 * y -1 >= z)
    """

    def __init__(self, expr: LinearExprT, bounds: Sequence[int]):
        self.__expr: LinearExprT = expr
        self.__bounds: Sequence[int] = bounds

    def __str__(self):
        if len(self.__bounds) == 2:
            lb, ub = self.__bounds
            if lb > INT_MIN and ub < INT_MAX:
                if lb == ub:
                    return str(self.__expr) + ' == ' + str(lb)
                else:
                    return str(lb) + ' <= ' + str(self.__expr) + ' <= ' + str(ub)
            elif lb > INT_MIN:
                return str(self.__expr) + ' >= ' + str(lb)
            elif ub < INT_MAX:
                return str(self.__expr) + ' <= ' + str(ub)
            else:
                return 'True (unbounded expr ' + str(self.__expr) + ')'
        elif len(self.__bounds) == 4 and self.__bounds[0] == INT_MIN and (self.__bounds[1] + 2 == self.__bounds[2]) and (self.__bounds[3] == INT_MAX):
            return str(self.__expr) + ' != ' + str(self.__bounds[1] + 1)
        else:
            return str(self.__expr) + ' in [' + display_bounds(self.__bounds) + ']'

    def expression(self) -> LinearExprT:
        return self.__expr

    def bounds(self) -> Sequence[int]:
        return self.__bounds

    def __bool__(self) -> bool:
        expr = self.__expr
        if isinstance(expr, LinearExpr):
            coeffs_map, constant = expr.get_integer_var_value_map()
            all_coeffs = set(coeffs_map.values())
            same_var = set([0])
            eq_bounds = [0, 0]
            different_vars = set([-1, 1])
            ne_bounds = [INT_MIN, -1, 1, INT_MAX]
            if len(coeffs_map) == 1 and all_coeffs == same_var and (constant == 0) and (self.__bounds == eq_bounds or self.__bounds == ne_bounds):
                return self.__bounds == eq_bounds
            if len(coeffs_map) == 2 and all_coeffs == different_vars and (constant == 0) and (self.__bounds == eq_bounds or self.__bounds == ne_bounds):
                return self.__bounds == ne_bounds
        raise NotImplementedError(f'Evaluating a BoundedLinearExpression "{self}" as a Boolean value' + ' is not supported.')