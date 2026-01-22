from typing import Any, TypeVar, Union, Optional
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs, apply_unitaries
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _strat_unitary_from_decompose(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _decompose_ method."""
    operations, qubits, val_qid_shape = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented
    all_qubits = frozenset((q for op in operations for q in op.qubits))
    work_qubits = frozenset(qubits)
    ancillas = tuple(sorted(all_qubits.difference(work_qubits)))
    ordered_qubits = ancillas + tuple(qubits)
    val_qid_shape = qid_shape_protocol.qid_shape(ancillas) + val_qid_shape
    result = apply_unitaries(operations, ordered_qubits, ApplyUnitaryArgs.for_unitary(qid_shape=val_qid_shape), None)
    if result is None:
        return None
    state_len = np.prod(val_qid_shape, dtype=np.int64)
    result = result.reshape((state_len, state_len))
    work_state_len = np.prod(val_qid_shape[len(ancillas):], dtype=np.int64)
    return result[:work_state_len, :work_state_len]