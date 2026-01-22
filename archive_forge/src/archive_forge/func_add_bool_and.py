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
def add_bool_and(self, *literals):
    """Adds `And(literals) == true`."""
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.bool_and.literals.extend([self.get_or_make_boolean_index(x) for x in expand_generator_or_tuple(literals)])
    return ct