import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
def _strat_apply_unitary_from_decompose(val: Any, args: ApplyUnitaryArgs) -> Optional[np.ndarray]:
    operations, qubits, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented
    all_qubits = frozenset([q for op in operations for q in op.qubits])
    ancilla = tuple(sorted(all_qubits.difference(qubits)))
    if not len(ancilla):
        return apply_unitaries(operations, qubits, args, None)
    ordered_qubits = ancilla + tuple(qubits)
    all_qid_shapes = qid_shape_protocol.qid_shape(ordered_qubits)
    result = apply_unitaries(operations, ordered_qubits, ApplyUnitaryArgs.for_unitary(qid_shape=all_qid_shapes), None)
    if result is None or result is NotImplemented:
        return result
    result = result.reshape((np.prod(all_qid_shapes, dtype=np.int64), -1))
    val_qid_shape = qid_shape_protocol.qid_shape(qubits)
    state_vec_length = np.prod(val_qid_shape, dtype=np.int64)
    result = result[:state_vec_length, :state_vec_length]
    return _apply_unitary_from_matrix(result, val, args)