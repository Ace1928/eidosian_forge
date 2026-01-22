import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _strat_apply_unitary_from_unitary(unitary_value: Any, args: ApplyUnitaryArgs) -> Optional[np.ndarray]:
    method = getattr(unitary_value, '_unitary_', None)
    if method is None:
        return NotImplemented
    matrix = method()
    if matrix is NotImplemented or matrix is None:
        return matrix
    return _apply_unitary_from_matrix(matrix, unitary_value, args)