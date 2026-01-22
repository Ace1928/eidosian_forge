import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def _get_target_name(op):
    """Get the name for the target operation. If the operation is not controlled, this is
    simplify the operation's name.
    """
    _control_base_map = {'CNOT': 'PauliX', 'CZ': 'PauliZ', 'CCZ': 'PauliZ', 'CY': 'PauliY', 'CH': 'Hadamard', 'CSWAP': 'SWAP', 'Toffoli': 'PauliX', 'ControlledPhaseShift': 'PhaseShift', 'CRX': 'RX', 'CRY': 'RY', 'CRZ': 'RZ', 'CRot': 'Rot', 'MultiControlledX': 'PauliX'}
    if op.name in _control_base_map:
        return _control_base_map[op.name]
    if isinstance(op, qml.ops.op_math.Controlled):
        return op.base.name
    return op.name