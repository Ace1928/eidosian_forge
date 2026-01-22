from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
class CCXPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """A Toffoli (doubly-controlled-NOT) that can be raised to a power.

    The unitary matrix of `CCX**t` is an 8x8 identity except the bottom right
    2x2 area is the matrix of `X**t`:

    $$
    \\begin{bmatrix}
        1 & & & & & & & \\\\
        & 1 & & & & & & \\\\
        & & 1 & & & & & \\\\
        & & & 1 & & & & \\\\
        & & & & 1 & & & \\\\
        & & & & & 1 & & \\\\
        & & & & & & e^{i \\pi t /2} \\cos(\\pi t) & -i e^{i \\pi t /2} \\sin(\\pi t) \\\\
        & & & & & & -i e^{i \\pi t /2} \\sin(\\pi t) & e^{i \\pi t /2} \\cos(\\pi t)
    \\end{bmatrix}
    $$
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, linalg.block_diag(np.diag([1, 1, 1, 1, 1, 1]), np.array([[0.5, 0.5], [0.5, 0.5]]))), (1, linalg.block_diag(np.diag([0, 0, 0, 0, 0, 0]), np.array([[0.5, -0.5], [-0.5, 0.5]])))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        z_phase = 1j ** self._exponent
        c = -1j * z_phase * np.sin(np.pi * self._exponent / 2) / 4
        return value.LinearDict({'III': global_phase * (1 - c), 'IIX': global_phase * c, 'IZI': global_phase * c, 'ZII': global_phase * c, 'ZZI': global_phase * -c, 'ZIX': global_phase * -c, 'IZX': global_phase * -c, 'ZZX': global_phase * c})

    def qubit_index_to_equivalence_group_key(self, index):
        return index < 2

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return protocols.apply_unitary(controlled_gate.ControlledGate(controlled_gate.ControlledGate(pauli_gates.X ** self.exponent)), protocols.ApplyUnitaryArgs(args.target_tensor, args.available_buffer, args.axes), default=NotImplemented)

    def _decompose_(self, qubits):
        c1, c2, t = qubits
        yield common_gates.H(t)
        yield CCZPowGate(exponent=self._exponent, global_shift=self.global_shift).on(c1, c2, t)
        yield common_gates.H(t)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(('@', '@', 'X'), exponent=self._diagram_exponent(args), exponent_qubit_index=2)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None
        args.validate_version('2.0')
        return args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.TOFFOLI'
            return f'(cirq.TOFFOLI**{proper_repr(self._exponent)})'
        return f'cirq.CCXPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'TOFFOLI'
        return f'TOFFOLI**{self._exponent}'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> raw_types.Gate:
        """Returns a controlled `XPowGate` with two additional controls.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = XPowGate`.
        """
        if num_controls == 0:
            return self
        sub_gate: 'cirq.Gate' = self
        if self._global_shift == 0:
            sub_gate = controlled_gate.ControlledGate(common_gates.XPowGate(exponent=self._exponent), num_controls=2)
        return controlled_gate.ControlledGate(sub_gate, num_controls=num_controls, control_values=control_values, control_qid_shape=control_qid_shape)