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
def add_allowed_assignments(self, variables: Sequence[VariableT], tuples_list: Iterable[Sequence[IntegralT]]) -> Constraint:
    """Adds AllowedAssignments(variables, tuples_list).

        An AllowedAssignments constraint is a constraint on an array of variables,
        which requires that when all variables are assigned values, the resulting
        array equals one of the  tuples in `tuple_list`.

        Args:
          variables: A list of variables.
          tuples_list: A list of admissible tuples. Each tuple must have the same
            length as the variables, and the ith value of a tuple corresponds to the
            ith variable.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          TypeError: If a tuple does not have the same size as the list of
              variables.
          ValueError: If the array of variables is empty.
        """
    if not variables:
        raise ValueError('add_allowed_assignments expects a non-empty variables array')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.table.vars.extend([self.get_or_make_index(x) for x in variables])
    arity = len(variables)
    for t in tuples_list:
        if len(t) != arity:
            raise TypeError('Tuple ' + str(t) + ' has the wrong arity')
        ar = []
        for v in t:
            ar.append(cmh.assert_is_int64(v))
        model_ct.table.values.extend(ar)
    return ct