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
class IntVar(LinearExpr):
    """An integer variable.

    An IntVar is an object that can take on any integer value within defined
    ranges. Variables appear in constraint like:

        x + y >= 5
        AllDifferent([x, y, z])

    Solving a model is equivalent to finding, for each variable, a single value
    from the set of initial values (called the initial domain), such that the
    model is feasible, or optimal if you provided an objective function.
    """

    def __init__(self, model: cp_model_pb2.CpModelProto, domain: Union[int, Domain], name: Optional[str]):
        """See CpModel.new_int_var below."""
        self.__negation: Optional[_NotBooleanVariable] = None
        if isinstance(domain, numbers.Integral) and name is None:
            self.__index: int = int(domain)
            self.__var: cp_model_pb2.IntegerVariableProto = model.variables[domain]
        else:
            self.__index: int = len(model.variables)
            self.__var: cp_model_pb2.IntegerVariableProto = model.variables.add()
            self.__var.domain.extend(cast(Domain, domain).flattened_intervals())
            self.__var.name = name

    @property
    def index(self) -> int:
        """Returns the index of the variable in the model."""
        return self.__index

    @property
    def proto(self) -> cp_model_pb2.IntegerVariableProto:
        """Returns the variable protobuf."""
        return self.__var

    def is_equal_to(self, other: Any) -> bool:
        """Returns true if self == other in the python sense."""
        if not isinstance(other, IntVar):
            return False
        return self.index == other.index

    def __str__(self) -> str:
        if not self.__var.name:
            if len(self.__var.domain) == 2 and self.__var.domain[0] == self.__var.domain[1]:
                return str(self.__var.domain[0])
            else:
                return 'unnamed_var_%i' % self.__index
        return self.__var.name

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__var.name, display_bounds(self.__var.domain))

    @property
    def name(self) -> str:
        if not self.__var or not self.__var.name:
            return ''
        return self.__var.name

    def negated(self) -> '_NotBooleanVariable':
        """Returns the negation of a Boolean variable.

        This method implements the logical negation of a Boolean variable.
        It is only valid if the variable has a Boolean domain (0 or 1).

        Note that this method is nilpotent: `x.negated().negated() == x`.
        """
        for bound in self.__var.domain:
            if bound < 0 or bound > 1:
                raise TypeError(f'cannot call negated on a non boolean variable: {self}')
        if self.__negation is None:
            self.__negation = _NotBooleanVariable(self)
        return self.__negation

    def __invert__(self) -> '_NotBooleanVariable':
        """Returns the logical negation of a Boolean variable."""
        return self.negated()
    Not = negated

    def Name(self) -> str:
        return self.name

    def Proto(self) -> cp_model_pb2.IntegerVariableProto:
        return self.proto

    def Index(self) -> int:
        return self.index