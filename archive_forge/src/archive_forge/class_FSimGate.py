import cmath
import math
from typing import AbstractSet, Any, Dict, Optional, Tuple
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import gate_features, raw_types
@value.value_equality(approximate=True)
class FSimGate(gate_features.InterchangeableQubitsGate, raw_types.Gate):
    """Fermionic simulation gate.

    Contains all two qubit interactions that preserve excitations, up to
    single-qubit rotations and global phase.

    The unitary matrix of this gate is:

    $$
    \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & a & b & 0 \\\\
        0 & b & a & 0 \\\\
        0 & 0 & 0 & c
    \\end{bmatrix}
    $$

    where:

    $$
    a = \\cos(\\theta)
    $$

    $$
    b = -i \\sin(\\theta)
    $$

    $$
    c = e^{-i \\phi}
    $$

    Note the difference in sign conventions between FSimGate and the
    ISWAP and CZPowGate:

    FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π)
    """

    def __init__(self, theta: 'cirq.TParamVal', phi: 'cirq.TParamVal') -> None:
        """Inits FSimGate.

        Args:
            theta: Swap angle on the ``|01⟩`` ``|10⟩`` subspace, in radians.
                Determined by the strength and duration of the XX+YY
                interaction. Note: uses opposite sign convention to the
                iSWAP gate. Maximum strength (full iswap) is at pi/2.
            phi: Controlled phase angle, in radians. Determines how much the
                ``|11⟩`` state is phased. Note: uses opposite sign convention to
                the CZPowGate. Maximum strength (full cz) is at pi.
        """
        self._theta = _canonicalize(theta)
        self._phi = _canonicalize(phi)

    @property
    def theta(self) -> 'cirq.TParamVal':
        return self._theta

    @property
    def phi(self) -> 'cirq.TParamVal':
        return self._phi

    def _num_qubits_(self) -> int:
        return 2

    def _value_equality_values_(self) -> Any:
        return (self.theta, self.phi)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.theta) or cirq.is_parameterized(self.phi)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.theta) | cirq.parameter_names(self.phi)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self) -> Optional[np.ndarray]:
        if self._is_parameterized_():
            return None
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        return np.array([[1, 0, 0, 0], [0, a, b, 0], [0, b, a, 0], [0, 0, 0, c]])

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if protocols.is_parameterized(self):
            return NotImplemented
        a = math.cos(self.theta)
        b = -1j * math.sin(self.theta)
        c = cmath.exp(-1j * self.phi)
        return value.LinearDict({'II': (1 + c) / 4 + a / 2, 'IZ': (1 - c) / 4, 'ZI': (1 - c) / 4, 'ZZ': (1 + c) / 4 - a / 2, 'XX': b / 2, 'YY': b / 2})

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'cirq.FSimGate':
        return FSimGate(protocols.resolve_parameters(self.theta, resolver, recursive), protocols.resolve_parameters(self.phi, resolver, recursive))

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        if cirq.is_parameterized(self):
            return None
        if self.theta != 0:
            inner_matrix = protocols.unitary(cirq.rx(2 * self.theta))
            oi = args.subspace_index(1)
            io = args.subspace_index(2)
            out = cirq.apply_matrix_to_slices(args.target_tensor, inner_matrix, slices=[oi, io], out=args.available_buffer)
        else:
            out = args.target_tensor
        if self.phi != 0:
            ii = args.subspace_index(3)
            out[ii] *= cmath.exp(-1j * self.phi)
        return out

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        a, b = qubits
        xx = cirq.XXPowGate(exponent=self.theta / np.pi, global_shift=-0.5)
        yy = cirq.YYPowGate(exponent=self.theta / np.pi, global_shift=-0.5)
        yield xx(a, b)
        yield yy(a, b)
        yield (cirq.CZ(a, b) ** (-self.phi / np.pi))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        t = args.format_radians(self.theta)
        p = args.format_radians(self.phi)
        return (f'FSim({t}, {p})', f'FSim({t}, {p})')

    def __pow__(self, power) -> 'FSimGate':
        return FSimGate(cirq.mul(self.theta, power), cirq.mul(self.phi, power))

    def __repr__(self) -> str:
        t = proper_repr(self.theta)
        p = proper_repr(self.phi)
        return f'cirq.FSimGate(theta={t}, phi={p})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['theta', 'phi'])