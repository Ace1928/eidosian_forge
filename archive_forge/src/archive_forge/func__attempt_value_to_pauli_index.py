import abc
import numbers
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, linalg, value
from cirq._compat import proper_repr
from cirq.ops import raw_types, identity, pauli_gates, global_phase_op, pauli_string
from cirq.type_workarounds import NotImplementedType
def _attempt_value_to_pauli_index(v: 'cirq.Operation') -> Optional[Tuple[int, int]]:
    if not isinstance(v, raw_types.Operation):
        return None
    if not isinstance(v.gate, pauli_gates.Pauli):
        return None
    q = v.qubits[0]
    from cirq import devices
    if not isinstance(q, devices.LineQubit):
        raise ValueError(f'Got a Pauli operation, but it was applied to a qubit type other than `cirq.LineQubit` so its dense index is ambiguous.\nv={repr(v)}.')
    return (pauli_string.PAULI_GATE_LIKE_TO_INDEX_MAP[v.gate], q.x)