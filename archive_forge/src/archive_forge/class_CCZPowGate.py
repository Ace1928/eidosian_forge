from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
class CCZPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """A doubly-controlled-Z that can be raised to a power.

    The unitary matrix of `CCZ**t` is (empty elements are $0$):

    $$
    \\begin{bmatrix}
        1 & & & & & & & \\\\
        & 1 & & & & & & \\\\
        & & 1 & & & & & \\\\
        & & & 1 & & & & \\\\
        & & & & 1 & & & \\\\
        & & & & & 1 & & \\\\
        & & & & & & 1 & \\\\
        & & & & & & & e^{i \\pi t}
    \\end{bmatrix}
    $$
    """

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 1, 1, 1, 1, 1, 1, 0])), (1, np.diag([0, 0, 0, 0, 0, 0, 0, 1]))]

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
        return value.LinearDict({'III': global_phase * (1 - c), 'IIZ': global_phase * c, 'IZI': global_phase * c, 'ZII': global_phase * c, 'ZZI': global_phase * -c, 'ZIZ': global_phase * -c, 'IZZ': global_phase * -c, 'ZZZ': global_phase * c})

    def _decompose_(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───p───@──────────────@───────@──────────@──────────
                  │              │       │          │
        1: ───p───X───@───p^-1───X───@───X──────@───X──────@───
                      │              │          │          │
        2: ───p───────X───p──────────X───p^-1───X───p^-1───X───

        where p = T**self._exponent
        """
        a, b, c = qubits
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = (c, b)
            elif not b.is_adjacent(c):
                a, b = (b, a)
        p = common_gates.T ** self._exponent
        sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
        global_phase = 1j ** (2 * self.global_shift * self._exponent)
        global_phase = complex(global_phase) if protocols.is_parameterized(global_phase) and global_phase.is_complex else global_phase
        global_phase_operation = [global_phase_op.global_phase_operation(global_phase)] if protocols.is_parameterized(global_phase) or abs(global_phase - 1.0) > 0 else []
        return global_phase_operation + [p(a), p(b), p(c), sweep_abc, p(b) ** (-1), p(c), sweep_abc, p(c) ** (-1), sweep_abc, p(c) ** (-1), sweep_abc]

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        ooo = args.subspace_index(7)
        args.target_tensor[ooo] *= np.exp(1j * self.exponent * np.pi)
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(('@', '@', '@'), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None
        args.validate_version('2.0')
        lines = [args.format('h {0};\n', qubits[2]), args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2]), args.format('h {0};\n', qubits[2])]
        return ''.join(lines)

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CCZ'
            return f'(cirq.CCZ**{proper_repr(self._exponent)})'
        return f'cirq.CCZPowGate(exponent={proper_repr(self._exponent)}, global_shift={self._global_shift!r})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CCZ'
        return f'CCZ**{self._exponent}'

    def _num_qubits_(self) -> int:
        return 3

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> raw_types.Gate:
        """Returns a controlled `ZPowGate` with two additional controls.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = ZPowGate`.
        """
        if num_controls == 0:
            return self
        sub_gate: 'cirq.Gate' = self
        if self._global_shift == 0:
            sub_gate = controlled_gate.ControlledGate(common_gates.ZPowGate(exponent=self._exponent), num_controls=2)
        return controlled_gate.ControlledGate(sub_gate, num_controls=num_controls, control_values=control_values, control_qid_shape=control_qid_shape)