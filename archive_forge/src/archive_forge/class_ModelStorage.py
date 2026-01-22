import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
class ModelStorage(abc.ABC):
    """An interface for in memory storage of an optimization model.

    Most users should not use this class directly and use Model defined in
    model.py.

    Stores an mixed integer programming problem of the form:

    {max/min} c*x + d
    s.t.  lb_c <= A * x <= ub_c
          lb_v <=     x <= ub_v
                      x_i integer for i in I

    where x is a vector of n decision variables, d is a number, lb_v, ub_v, and c
    are vectors of n numbers, lb_c and ub_c are vectors of m numbers, A is a
    m by n matrix, and I is a subset of {1,..., n}.

    Each of the n variables and m constraints have an integer id that you use to
    get/set the problem data (c, A, lb_c etc.). Ids begin at zero and increase
    sequentially. They are not reused after deletion. Note that if a variable is
    deleted, your model has nonconsecutive variable ids.

    For all methods taking an id (e.g. set_variable_lb), providing a bad id
    (including the id of a deleted variable) will raise a BadVariableIdError or
    BadLinearConstraintIdError. Further, the ModelStorage instance is assumed to
    be in a bad state after any such error and there are no guarantees on further
    interactions.

    All implementations must have a constructor taking a str argument for the
    model name with a default value of the empty string.

    Any ModelStorage can be exported to model_pb2.ModelProto, the format consumed
    by MathOpt solvers. Changes to a model can be exported to a
    model_update_pb2.ModelUpdateProto with an UpdateTracker, see the UpdateTracker
    documentation for details.

    When solving this optimization problem we will additionally require that:
      * No numbers are NaN,
      * c, d, and A are all finite,
      * lb_c and lb_v are not +inf,
      * ub_c and ub_v are not -inf,
    but those assumptions are not checked or enforced here (NaNs and infinite
    values can be used anywhere).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def add_variable(self, lb: float, ub: float, is_integer: bool, name: str) -> int:
        pass

    @abc.abstractmethod
    def delete_variable(self, variable_id: int) -> None:
        pass

    @abc.abstractmethod
    def variable_exists(self, variable_id: int) -> bool:
        pass

    @abc.abstractmethod
    def next_variable_id(self) -> int:
        pass

    @abc.abstractmethod
    def set_variable_lb(self, variable_id: int, lb: float) -> None:
        pass

    @abc.abstractmethod
    def set_variable_ub(self, variable_id: int, ub: float) -> None:
        pass

    @abc.abstractmethod
    def set_variable_is_integer(self, variable_id: int, is_integer: bool) -> None:
        pass

    @abc.abstractmethod
    def get_variable_lb(self, variable_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_variable_ub(self, variable_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_variable_is_integer(self, variable_id: int) -> bool:
        pass

    @abc.abstractmethod
    def get_variable_name(self, variable_id: int) -> str:
        pass

    @abc.abstractmethod
    def get_variables(self) -> Iterator[int]:
        """Yields the variable ids in order of creation."""
        pass

    @abc.abstractmethod
    def add_linear_constraint(self, lb: float, ub: float, name: str) -> int:
        pass

    @abc.abstractmethod
    def delete_linear_constraint(self, linear_constraint_id: int) -> None:
        pass

    @abc.abstractmethod
    def linear_constraint_exists(self, linear_constraint_id: int) -> bool:
        pass

    @abc.abstractmethod
    def next_linear_constraint_id(self) -> int:
        pass

    @abc.abstractmethod
    def set_linear_constraint_lb(self, linear_constraint_id: int, lb: float) -> None:
        pass

    @abc.abstractmethod
    def set_linear_constraint_ub(self, linear_constraint_id: int, ub: float) -> None:
        pass

    @abc.abstractmethod
    def get_linear_constraint_lb(self, linear_constraint_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_linear_constraint_ub(self, linear_constraint_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_linear_constraint_name(self, linear_constraint_id: int) -> str:
        pass

    @abc.abstractmethod
    def get_linear_constraints(self) -> Iterator[int]:
        """Yields the linear constraint ids in order of creation."""
        pass

    @abc.abstractmethod
    def set_linear_constraint_coefficient(self, linear_constraint_id: int, variable_id: int, lb: float) -> None:
        pass

    @abc.abstractmethod
    def get_linear_constraint_coefficient(self, linear_constraint_id: int, variable_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_linear_constraints_with_variable(self, variable_id: int) -> Iterator[int]:
        """Yields the linear constraints with nonzero coefficient for a variable in undefined order."""
        pass

    @abc.abstractmethod
    def get_variables_for_linear_constraint(self, linear_constraint_id: int) -> Iterator[int]:
        """Yields the variables with nonzero coefficient in a linear constraint in undefined order."""
        pass

    @abc.abstractmethod
    def get_linear_constraint_matrix_entries(self) -> Iterator[LinearConstraintMatrixIdEntry]:
        """Yields the nonzero elements of the linear constraint matrix in undefined order."""
        pass

    @abc.abstractmethod
    def clear_objective(self) -> None:
        """Clears objective coefficients and offset. Does not change direction."""

    @abc.abstractmethod
    def set_linear_objective_coefficient(self, variable_id: int, value: float) -> None:
        pass

    @abc.abstractmethod
    def get_linear_objective_coefficient(self, variable_id: int) -> float:
        pass

    @abc.abstractmethod
    def get_linear_objective_coefficients(self) -> Iterator[LinearObjectiveEntry]:
        """Yields the nonzero linear objective terms in undefined order."""
        pass

    @abc.abstractmethod
    def set_quadratic_objective_coefficient(self, first_variable_id: int, second_variable_id: int, value: float) -> None:
        """Sets the objective coefficient for the product of two variables.

        The ordering of the input variables does not matter.

        Args:
          first_variable_id: The first variable in the product.
          second_variable_id: The second variable in the product.
          value: The value of the coefficient.

        Raises:
          BadVariableIdError if first_variable_id or second_variable_id are not in
          the model.
        """

    @abc.abstractmethod
    def get_quadratic_objective_coefficient(self, first_variable_id: int, second_variable_id: int) -> float:
        """Gets the objective coefficient for the product of two variables.

        The ordering of the input variables does not matter.

        Args:
          first_variable_id: The first variable in the product.
          second_variable_id: The second variable in the product.

        Raises:
          BadVariableIdError if first_variable_id or second_variable_id are not in
          the model.

        Returns:
          The value of the coefficient.
        """

    @abc.abstractmethod
    def get_quadratic_objective_coefficients(self) -> Iterator[QuadraticEntry]:
        """Yields the nonzero quadratic objective terms in undefined order."""

    @abc.abstractmethod
    def get_quadratic_objective_adjacent_variables(self, variable_id: int) -> Iterator[int]:
        """Yields the variables multiplying a variable in the objective function.

        Variables are returned in an unspecified order.

        For example, if variables x and y have ids 0 and 1 respectively, and the
        quadratic portion of the objective is x^2 + 2 x*y, then
        get_quadratic_objective_adjacent_variables(0) = (0, 1).

        Args:
          variable_id: Function yields the variables multiplying variable_id in the
            objective function.

        Yields:
          The variables multiplying variable_id in the objective function.

        Raises:
          BadVariableIdError if variable_id is not in the model.
        """

    @abc.abstractmethod
    def set_is_maximize(self, is_maximize: bool) -> None:
        pass

    @abc.abstractmethod
    def get_is_maximize(self) -> bool:
        pass

    @abc.abstractmethod
    def set_objective_offset(self, offset: float) -> None:
        pass

    @abc.abstractmethod
    def get_objective_offset(self) -> float:
        pass

    @abc.abstractmethod
    def export_model(self) -> model_pb2.ModelProto:
        pass

    @abc.abstractmethod
    def add_update_tracker(self) -> StorageUpdateTracker:
        """Creates a StorageUpdateTracker registered with self to view model changes."""
        pass

    @abc.abstractmethod
    def remove_update_tracker(self, tracker: StorageUpdateTracker):
        """Stops tracker from getting updates on model changes in self.

        An error will be raised if tracker is not a StorageUpdateTracker created by
        this Model that has not previously been removed.

        Using an UpdateTracker (via checkpoint or export_update) after it has been
        removed will result in an error.

        Args:
          tracker: The StorageUpdateTracker to unregister.

        Raises:
          KeyError: The tracker was created by another model or was already removed.
        """
        pass