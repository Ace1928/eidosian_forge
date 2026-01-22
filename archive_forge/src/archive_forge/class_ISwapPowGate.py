from typing import Optional, Tuple, TYPE_CHECKING, List
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import common_gates, gate_features, eigen_gate
class ISwapPowGate(gate_features.InterchangeableQubitsGate, eigen_gate.EigenGate):
    """Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

    When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by i. More
    generally, this gate's matrix is defined as follows:

        ISWAP**t ≡ exp(+i π t (X⊗X + Y⊗Y) / 4)

    which is given by the matrix:

    $$
    \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & c & i s & 0 \\\\
        0 & i s & c & 0 \\\\
        0 & 0 & 0 & 1
    \\end{bmatrix}
    $$

    where

    $$
    c = \\cos\\left(\\frac{\\pi t}{2}\\right)
    $$
    $$
    s = \\sin\\left(\\frac{\\pi t}{2}\\right)
    $$

    `cirq.ISWAP`, the swap gate that applies i to the |01⟩ and |10⟩ states,
    is an instance of this gate at exponent=1.

    References:
        "What is the matrix of the iSwap gate?"
        https://quantumcomputing.stackexchange.com/questions/2594/
    """

    def _num_qubits_(self) -> int:
        return 2

    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [(0, np.diag([1, 0, 0, 1])), (+0.5, np.array([[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]])), (-0.5, np.array([[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield common_gates.CNOT(a, b)
        yield common_gates.H(a)
        yield common_gates.CNOT(b, a)
        yield common_gates.ZPowGate(exponent=self._exponent / 2, global_shift=self.global_shift).on(a)
        yield common_gates.CNOT(b, a)
        yield common_gates.ZPowGate(exponent=-self._exponent / 2, global_shift=-self.global_shift).on(a)
        yield common_gates.H(a)
        yield common_gates.CNOT(a, b)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if self._exponent != 1:
            return NotImplemented
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        args.target_tensor[zo] *= 1j
        args.target_tensor[oz] *= 1j
        p = 1j ** (2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        global_phase = 1j ** (2 * self._exponent * self._global_shift)
        angle = np.pi * self._exponent / 4
        c, s = (np.cos(angle), np.sin(angle))
        return value.LinearDict({'II': global_phase * c * c, 'XX': global_phase * c * s * 1j, 'YY': global_phase * s * c * 1j, 'ZZ': global_phase * s * s})

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('iSwap', 'iSwap'), exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ISWAP'
        if self._exponent == -1:
            return 'ISWAP_INV'
        return f'ISWAP**{self._exponent}'

    def __repr__(self) -> str:
        e = proper_repr(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ISWAP'
            if self._exponent == -1:
                return 'cirq.ISWAP_INV'
            return f'(cirq.ISWAP**{e})'
        return f'cirq.ISwapPowGate(exponent={e}, global_shift={self._global_shift!r})'