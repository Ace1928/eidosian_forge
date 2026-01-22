from typing import Any, TypeVar, Optional
import numpy as np
from typing_extensions import Protocol
from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
def _strat_has_unitary_from_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _unitary_ method."""
    getter = getattr(val, '_unitary_', None)
    if getter is None:
        return None
    result = getter()
    return result is not NotImplemented and result is not None