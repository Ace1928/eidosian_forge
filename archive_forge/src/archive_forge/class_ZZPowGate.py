from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
@value.value_equality
class ZZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """The Z-parity gate, possibly raised to a power.

    The ZZ**t gate implements the following unitary:

    $$
    (Z \\otimes Z)^t = \\begin{bmatrix}
                      1 & & & \\\\
                      & e^{i \\pi t} & & \\\\
                      & & e^{i \\pi t} & \\\\
                      & & & 1
                      \\end{bmatrix}
    $$
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits):
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[0])
        yield common_gates.ZPowGate(exponent=self.exponent)(qubits[1])
        yield common_gates.CZPowGate(exponent=-2 * self.exponent, global_shift=-self.global_shift / 2)(qubits[0], qubits[1])

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

    def _has_stabilizer_effect_(self) -> bool:
        return self.exponent % 2 in (0, 0.5, 1, 1.5)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0, 0, 1])), (1, np.diag([0, 1, 1, 0]))]

    def _eigen_shifts(self):
        return [0, 1]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('ZZ', 'ZZ'), exponent=self._diagram_exponent(args))

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if protocols.is_parameterized(self):
            return None
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        if global_phase != 1:
            args.target_tensor *= global_phase
        relative_phase = 1j ** (2 * self.exponent)
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        args.target_tensor[oz] *= relative_phase
        args.target_tensor[zo] *= relative_phase
        return args.target_tensor

    def _phase_by_(self, phase_turns: float, qubit_index: int) -> 'ZZPowGate':
        return self

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZZ'
        return f'ZZ**{self._exponent}'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZZ'
            return f'(cirq.ZZ**{proper_repr(self._exponent)})'
        return f'cirq.ZZPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'