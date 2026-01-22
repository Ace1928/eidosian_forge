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
def SearchForAllSolutions(self, model: CpModel, callback: 'CpSolverSolutionCallback') -> cp_model_pb2.CpSolverStatus:
    """DEPRECATED Use solve() with the right parameter.

        Search for all solutions of a satisfiability problem.

        This method searches for all feasible solutions of a given model.
        Then it feeds the solution to the callback.

        Note that the model cannot contain an objective.

        Args:
          model: The model to solve.
          callback: The callback that will be called at each solution.

        Returns:
          The status of the solve:

          * *FEASIBLE* if some solutions have been found
          * *INFEASIBLE* if the solver has proved there are no solution
          * *OPTIMAL* if all solutions have been found
        """
    warnings.warn('search_for_all_solutions is deprecated; use solve() with' + 'enumerate_all_solutions = True.', DeprecationWarning)
    if model.has_objective():
        raise TypeError('Search for all solutions is only defined on satisfiability problems')
    enumerate_all = self.parameters.enumerate_all_solutions
    self.parameters.enumerate_all_solutions = True
    self.solve(model, callback)
    self.parameters.enumerate_all_solutions = enumerate_all
    return self.__solution.status