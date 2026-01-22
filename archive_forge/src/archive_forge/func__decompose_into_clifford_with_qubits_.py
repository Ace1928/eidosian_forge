from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
def _decompose_into_clifford_with_qubits_(self, qubits: Sequence['cirq.Qid']) -> Sequence[Union['cirq.Operation', Sequence['cirq.Operation']]]:
    if not self._has_stabilizer_effect_():
        return NotImplemented
    if self.exponent % 2 == 0:
        return []
    if self.exponent % 2 == 1:
        return clifford_gate.SingleQubitCliffordGate.Z.on_each(*qubits)
    if self.exponent % 2 == 0.5:
        return [pauli_interaction_gate.PauliInteractionGate(pauli_gates.Z, False, pauli_gates.Z, False).on(*qubits), clifford_gate.SingleQubitCliffordGate.Z_sqrt.on_each(*qubits)]
    else:
        return [pauli_interaction_gate.PauliInteractionGate(pauli_gates.Z, False, pauli_gates.Z, False).on(*qubits), clifford_gate.SingleQubitCliffordGate.Z_nsqrt.on_each(*qubits)]