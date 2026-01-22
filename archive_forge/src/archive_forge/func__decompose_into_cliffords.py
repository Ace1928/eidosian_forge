import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _decompose_into_cliffords(op: 'cirq.Operation') -> List['cirq.Operation']:
    if isinstance(op.gate, global_phase_op.GlobalPhaseGate):
        return []
    if isinstance(op.gate, (clifford_gate.SingleQubitCliffordGate, pauli_interaction_gate.PauliInteractionGate)):
        return [op]
    v = getattr(op, '_decompose_into_clifford_', None)
    if v is not None:
        result = v()
        if result is not None and result is not NotImplemented:
            return list(op_tree.flatten_to_ops(result))
    decomposed = protocols.decompose_once(op, None)
    if decomposed is not None:
        return [out for sub_op in decomposed for out in _decompose_into_cliffords(sub_op)]
    raise TypeError(f'Operation is not a known Clifford and did not decompose into known Cliffords: {op!r}')