from typing import Any, TypeVar, Optional
import numpy as np
from typing_extensions import Protocol
from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
def _strat_has_unitary_from_apply_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _apply_unitary_ method."""
    method = getattr(val, '_apply_unitary_', None)
    if method is None:
        return None
    val_qid_shape = qid_shape_protocol.qid_shape(val, None)
    if val_qid_shape is None:
        return None
    state = qis.one_hot(shape=val_qid_shape, dtype=np.complex64)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(len(val_qid_shape))))
    if result is NotImplemented:
        return None
    return result is not None