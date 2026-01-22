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
def add_inverse(self, variables: Sequence[VariableT], inverse_variables: Sequence[VariableT]) -> Constraint:
    """Adds Inverse(variables, inverse_variables).

        An inverse constraint enforces that if `variables[i]` is assigned a value
        `j`, then `inverse_variables[j]` is assigned a value `i`. And vice versa.

        Args:
          variables: An array of integer variables.
          inverse_variables: An array of integer variables.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: if variables and inverse_variables have different lengths, or
              if they are empty.
        """
    if not variables or not inverse_variables:
        raise TypeError('The Inverse constraint does not accept empty arrays')
    if len(variables) != len(inverse_variables):
        raise TypeError('In the inverse constraint, the two array variables and inverse_variables must have the same length.')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.inverse.f_direct.extend([self.get_or_make_index(x) for x in variables])
    model_ct.inverse.f_inverse.extend([self.get_or_make_index(x) for x in inverse_variables])
    return ct