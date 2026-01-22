import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
class _BoundedLinearExpr(metaclass=abc.ABCMeta):
    """Interface for types that can build bounded linear (boolean) expressions.

    Classes derived from _BoundedLinearExpr are used to build linear constraints
    to be satisfied.

      * BoundedLinearExpression: a linear expression with upper and lower bounds.
      * VarEqVar: an equality comparison between two variables.
    """

    @abc.abstractmethod
    def _add_linear_constraint(self, helper: mbh.ModelBuilderHelper, name: str) -> 'LinearConstraint':
        """Creates a new linear constraint in the helper.

        Args:
          helper (mbh.ModelBuilderHelper): The helper to create the constraint.
          name (str): The name of the linear constraint.

        Returns:
          LinearConstraint: A reference to the linear constraint in the helper.
        """

    @abc.abstractmethod
    def _add_enforced_linear_constraint(self, helper: mbh.ModelBuilderHelper, var: Variable, value: bool, name: str) -> 'EnforcedLinearConstraint':
        """Creates a new enforced linear constraint in the helper.

        Args:
          helper (mbh.ModelBuilderHelper): The helper to create the constraint.
          var (Variable): The indicator variable of the constraint.
          value (bool): The indicator value of the constraint.
          name (str): The name of the linear constraint.

        Returns:
          Enforced LinearConstraint: A reference to the linear constraint in the
          helper.
        """