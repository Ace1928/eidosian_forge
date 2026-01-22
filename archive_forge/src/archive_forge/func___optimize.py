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
def __optimize(self, linear_expr: LinearExprT, maximize: bool) -> None:
    """Defines the objective."""
    self.helper.clear_objective()
    self.__helper.set_maximize(maximize)
    if mbn.is_a_number(linear_expr):
        self.helper.set_objective_offset(linear_expr)
    elif isinstance(linear_expr, Variable):
        self.helper.set_var_objective_coefficient(linear_expr.index, 1.0)
    elif isinstance(linear_expr, LinearExpr):
        flat_expr = _as_flat_linear_expression(linear_expr)
        self.helper.set_objective_offset(flat_expr._offset)
        self.helper.set_objective_coefficients(flat_expr._variable_indices, flat_expr._coefficients)
    else:
        raise TypeError(f'Not supported: Model.minimize/maximize({linear_expr})')