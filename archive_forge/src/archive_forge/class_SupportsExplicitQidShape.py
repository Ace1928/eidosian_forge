from typing import Any, Sequence, Tuple, TypeVar, Union
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import document, doc_private
from cirq.type_workarounds import NotImplementedType
class SupportsExplicitQidShape(Protocol):
    """A unitary, channel, mixture or other object that operates on a known
    number qubits/qudits/qids, each with a specific number of quantum levels."""

    @doc_private
    def _qid_shape_(self) -> Union[Tuple[int, ...], NotImplementedType]:
        """A tuple specifying the number of quantum levels of each qid this
        object operates on, e.g. (2, 2, 2) for a three-qubit gate.

        This method is used by the global `cirq.qid_shape` method (and by
        `cirq.num_qubits` if `_num_qubits_` is not defined). If this
        method is not present, or returns NotImplemented, it is assumed that the
        receiving object operates on qubits. (The ability to return
        NotImplemented is useful when a class cannot know if it has a shape
        until runtime.)

        The order of values in the tuple is always implicit with respect to the
        object being called. For example, for gates the tuple must be ordered
        with respect to the list of qubits that the gate is applied to. For
        operations, the tuple is ordered to match the list returned by its
        `qubits` attribute.

        Returns:
            The qid shape of this value, or NotImplemented if the shape is
            unknown.
        """