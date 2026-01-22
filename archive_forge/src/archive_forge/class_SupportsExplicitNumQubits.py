from typing import Any, Sequence, Tuple, TypeVar, Union
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import document, doc_private
from cirq.type_workarounds import NotImplementedType
class SupportsExplicitNumQubits(Protocol):
    """A unitary, channel, mixture or other object that operates on a known
    number of qubits."""

    @document
    def _num_qubits_(self) -> Union[int, NotImplementedType]:
        """The number of qubits, qudits, or qids this object operates on.

        This method is used by the global `cirq.num_qubits` method (and by
        `cirq.qid_shape` if `_qid_shape_` is not defined.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using the length of `_qid_shape_`.

        Returns:
            An integer specifying the number of qubits, qudits or qids.
        """