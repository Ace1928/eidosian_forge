import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def variable_status(self, variables=None):
    """The variable basis status associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        basis associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated basis.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            reduced costs to return. If not provided, variable_status() returns a
            dictionary with the reduced costs for all variables.

        Returns:
          The variable basis status associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated basis.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
    if not self.has_basis():
        raise ValueError(_NO_BASIS_ERROR)
    assert self.solutions[0].basis is not None
    if variables is None:
        return self.solutions[0].basis.variable_status
    if isinstance(variables, model.Variable):
        return self.solutions[0].basis.variable_status[variables]
    if isinstance(variables, Iterable):
        return [self.solutions[0].basis.variable_status[v] for v in variables]
    raise TypeError(f'unsupported type in argument for variable_status: {type(variables).__name__!r}')