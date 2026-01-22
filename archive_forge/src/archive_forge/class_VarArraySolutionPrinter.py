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
class VarArraySolutionPrinter(CpSolverSolutionCallback):
    """Print intermediate solutions (variable values, time)."""

    def __init__(self, variables: Sequence[IntVar]):
        CpSolverSolutionCallback.__init__(self)
        self.__variables: Sequence[IntVar] = variables
        self.__solution_count: int = 0
        self.__start_time: float = time.time()

    def on_solution_callback(self) -> None:
        """Called on each new solution."""
        current_time = time.time()
        print('Solution %i, time = %0.2f s' % (self.__solution_count, current_time - self.__start_time))
        for v in self.__variables:
            print('  %s = %i' % (v, self.value(v)), end=' ')
        print()
        self.__solution_count += 1

    @property
    def solution_count(self) -> int:
        """Returns the number of solutions found."""
        return self.__solution_count