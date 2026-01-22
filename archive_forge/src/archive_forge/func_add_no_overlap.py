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
def add_no_overlap(self, interval_vars: Iterable[IntervalVar]) -> Constraint:
    """Adds NoOverlap(interval_vars).

        A NoOverlap constraint ensures that all present intervals do not overlap
        in time.

        Args:
          interval_vars: The list of interval variables to constrain.

        Returns:
          An instance of the `Constraint` class.
        """
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.no_overlap.intervals.extend([self.get_interval_index(x) for x in interval_vars])
    return ct