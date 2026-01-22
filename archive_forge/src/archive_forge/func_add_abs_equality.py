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
def add_abs_equality(self, target: LinearExprT, expr: LinearExprT) -> Constraint:
    """Adds `target == Abs(expr)`."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.lin_max.exprs.append(self.parse_linear_expression(expr))
    model_ct.lin_max.exprs.append(self.parse_linear_expression(expr, True))
    model_ct.lin_max.target.CopyFrom(self.parse_linear_expression(target))
    return ct