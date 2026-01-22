from typing import Any, Optional
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq._import import LazyLoader
import cirq.protocols.unitary_protocol as unitary_protocol
import cirq.protocols.has_unitary_protocol as has_unitary_protocol
import cirq.protocols.qid_shape_protocol as qid_shape_protocol
import cirq.protocols.decompose_protocol as decompose_protocol
def _strat_has_stabilizer_effect_from_decompose(val: Any) -> Optional[bool]:
    qid_shape = qid_shape_protocol.qid_shape(val, default=None)
    if qid_shape is None or len(qid_shape) <= 3:
        return None
    decomposition = decompose_protocol.decompose_once(val, default=None)
    if decomposition is None:
        return None
    for op in decomposition:
        res = has_stabilizer_effect(op)
        if not res:
            return res
    return True