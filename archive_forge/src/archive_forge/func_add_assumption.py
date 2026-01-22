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
def add_assumption(self, lit: LiteralT) -> None:
    """Adds the literal to the model as assumptions."""
    self.__model.assumptions.append(self.get_or_make_boolean_index(lit))