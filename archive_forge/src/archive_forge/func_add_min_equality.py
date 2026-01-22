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
def add_min_equality(self, target: LinearExprT, exprs: Iterable[LinearExprT]) -> Constraint:
    """Adds `target == Min(exprs)`."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.lin_max.exprs.extend([self.parse_linear_expression(x, True) for x in exprs])
    model_ct.lin_max.target.CopyFrom(self.parse_linear_expression(target, True))
    return ct