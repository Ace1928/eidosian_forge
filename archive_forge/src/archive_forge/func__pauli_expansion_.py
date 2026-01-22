from typing import Any, TypeVar, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.linalg import operator_spaces
from cirq.protocols import qid_shape_protocol, unitary_protocol
@doc_private
def _pauli_expansion_(self) -> value.LinearDict[str]:
    """Efficiently obtains expansion of self in the Pauli basis.

        Returns:
            Linear dict keyed by name of Pauli basis element. The names
            consist of n captal letters from the set 'I', 'X', 'Y', 'Z'
            where n is the number of qubits. For example, 'II', 'IX' and
            'XY' are valid Pauli names in the two-qubit case.
        """