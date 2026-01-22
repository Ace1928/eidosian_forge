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
def add_multiplication_equality(self, target: LinearExprT, *expressions: Union[Iterable[LinearExprT], LinearExprT]) -> Constraint:
    """Adds `target == expressions[0] * .. * expressions[n]`."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.int_prod.exprs.extend([self.parse_linear_expression(expr) for expr in expand_generator_or_tuple(expressions)])
    model_ct.int_prod.target.CopyFrom(self.parse_linear_expression(target))
    return ct