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
def SolveWithSolutionCallback(self, model: CpModel, callback: 'CpSolverSolutionCallback') -> cp_model_pb2.CpSolverStatus:
    """DEPRECATED Use solve() with the callback argument."""
    warnings.warn('solve_with_solution_callback is deprecated; use solve() with' + 'the callback argument.', DeprecationWarning)
    return self.solve(model, callback)