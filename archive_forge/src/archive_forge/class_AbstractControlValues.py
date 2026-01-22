import abc
from typing import Collection, Tuple, TYPE_CHECKING, Any, Dict, Iterator, Optional, Sequence, Union
import itertools
from cirq import protocols, value, _compat
@value.value_equality
class AbstractControlValues(abc.ABC):
    """Abstract base class defining the API for control values.

    Control values define predicates on the state of one or more qubits. Predicates can be composed
    with logical OR to form a "sum", or with logical AND to form a "product". We provide two
    implementations: `SumOfProducts` which consists of one or more AND (product) clauses each of
    which applies to all N qubits, and `ProductOfSums` which consists of N OR (sum) clauses,
    each of which applies to one qubit.

    `cirq.ControlledGate` and `cirq.ControlledOperation` are useful to augment
    existing gates and operations to have one or more control qubits. For every
    control qubit, the set of integer values for which the control should be enabled
    is represented by one of the implementations of `cirq.AbstractControlValues`.

    Implementations of `cirq.AbstractControlValues` can use different internal
    representations to store control values, but they must satisfy the public API
    defined here and be immutable.
    """

    @abc.abstractmethod
    def validate(self, qid_shapes: Sequence[int]) -> None:
        """Validates that all control values for ith qubit are in range [0, qid_shaped[i])"""

    @abc.abstractmethod
    def expand(self) -> 'SumOfProducts':
        """Returns an expanded `cirq.SumOfProduct` representation of this control values."""

    @property
    @abc.abstractmethod
    def is_trivial(self) -> bool:
        """Returns True iff each controlled variable is activated only for value 1.

        This configuration is equivalent to `cirq.SumOfProducts(((1,) * num_controls))`
        and `cirq.ProductOfSums(((1,),) * num_controls)`
        """

    @abc.abstractmethod
    def _num_qubits_(self) -> int:
        """Returns the number of qubits for which control values are stored by this object."""

    @abc.abstractmethod
    def _json_dict_(self) -> Dict[str, Any]:
        """Returns a dictionary used for serializing this object."""

    @abc.abstractmethod
    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        """Returns information used to draw this object in circuit diagrams."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Iterator on internal representation of control values used by the derived classes.

        Note: Be careful that the terms iterated upon by this iterator will have different
        meaning based on the implementation. For example:
        >>> print(*cirq.ProductOfSums([(0, 1), (0,)]))
        (0, 1) (0,)
        >>> print(*cirq.SumOfProducts([(0, 0), (1, 0)]))
        (0, 0) (1, 0)
        """

    def _value_equality_values_(self) -> Any:
        return tuple((v for v in self.expand()))

    def __and__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Returns a cartesian product of all control values predicates in `self` x `other`.

        The `and` of two control values `cv1` and `cv2` represents a control value object
        acting on the union of qubits represented by `cv1` and `cv2`. For example:

        >>> cv1 = cirq.ProductOfSums([(0, 1), 2])
        >>> cv2 = cirq.SumOfProducts([[0, 0], [1, 1]])
        >>> assert cirq.num_qubits(cv1 & cv2) == cirq.num_qubits(cv1) + cirq.num_qubits(cv2)

        Args:
          other: An instance of `AbstractControlValues`.

        Returns:
          An instance of `AbstractControlValues` that represents the cartesian product of
          control values represented by `self` and `other`.
        """
        return SumOfProducts(tuple((x + y for x, y in itertools.product(self.expand(), other.expand()))))

    def __or__(self, other: 'AbstractControlValues') -> 'AbstractControlValues':
        """Returns a union of all control values predicates in `self` + `other`.

        Both `self` and `other` must represent control values for the same set of qubits and
        hence their `or` would also be a control value object acting on the same set of qubits.
        For example:

        >>> cv1 = cirq.ProductOfSums([(0, 1), 2])
        >>> cv2 = cirq.SumOfProducts([[0, 0], [1, 1]])
        >>> assert cirq.num_qubits(cv1 | cv2) == cirq.num_qubits(cv1) == cirq.num_qubits(cv2)

        Args:
          other: An instance of `AbstractControlValues`.

        Returns:
          An instance of `AbstractControlValues` that represents the union of control values
          represented by `self` and `other`.

        Raises:
            ValueError: If `cirq.num_qubits(self) != cirq.num_qubits(other)`.
        """
        if protocols.num_qubits(self) != protocols.num_qubits(other):
            raise ValueError(f'Control values {self} and {other} must act on equal number of qubits')
        return SumOfProducts((*self.expand(), *other.expand()))