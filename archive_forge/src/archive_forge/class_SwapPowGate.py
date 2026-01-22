from typing import Optional, Tuple, TYPE_CHECKING, List
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import common_gates, gate_features, eigen_gate
class SwapPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """The SWAP gate, possibly raised to a power. Exchanges qubits.

    SwapPowGate()**t = SwapPowGate(exponent=t) and acts on two qubits in the
    computational basis as the matrix:

    $$
    \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & g c & -i g s & 0 \\\\
        0 & -i g s & g c & 0 \\\\
        0 & 0 & 0 & 1
    \\end{bmatrix}
    $$

    where:

    $$
    c = \\cos\\left(\\frac{\\pi t}{2}\\right)
    $$
    $$
    s = \\sin\\left(\\frac{\\pi t}{2}\\right)
    $$
    $$
    g = e^{\\frac{i \\pi t}{2}}
    $$

    `cirq.SWAP`, the swap gate, is an instance of this gate at exponent=1.
    """

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_(self, qubits):
        """See base class."""
        a, b = qubits
        yield common_gates.CNOT(a, b)
        yield common_gates.CNotPowGate(exponent=self._exponent, global_shift=self.global_shift).on(b, a)
        yield common_gates.CNOT(a, b)

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.array([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]])), (1, np.array([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]]))]

    def _trace_distance_bound_(self) -> Optional[float]:
        if self._is_parameterized_():
            return None
        return abs(np.sin(self._exponent * 0.5 * np.pi))

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        if self._is_parameterized_():
            return None
        return self.exponent % 1 == 0

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        swap_phase = 1j ** self._exponent
        c = -1j * swap_phase * np.sin(np.pi * self._exponent / 2) / 2
        return value.LinearDict({'II': global_phase * (1 - c), 'XX': global_phase * c, 'YY': global_phase * c, 'ZZ': global_phase * c})

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(wire_symbols=('Swap', 'Swap'), exponent=self._diagram_exponent(args))
        return protocols.CircuitDiagramInfo(wire_symbols=('×', '×'), exponent=self._diagram_exponent(args))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self._exponent != 1:
            return None
        args.validate_version('2.0')
        return args.format('swap {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'SWAP'
        return f'SWAP**{self._exponent}'

    def __repr__(self) -> str:
        e = proper_repr(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.SWAP'
            return f'(cirq.SWAP**{e})'
        return f'cirq.SwapPowGate(exponent={e}, global_shift={self._global_shift!r})'