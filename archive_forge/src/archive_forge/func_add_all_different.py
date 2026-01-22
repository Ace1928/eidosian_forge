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
def add_all_different(self, *expressions):
    """Adds AllDifferent(expressions).

        This constraint forces all expressions to have different values.

        Args:
          *expressions: simple expressions of the form a * var + constant.

        Returns:
          An instance of the `Constraint` class.
        """
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    expanded = expand_generator_or_tuple(expressions)
    model_ct.all_diff.exprs.extend((self.parse_linear_expression(x) for x in expanded))
    return ct