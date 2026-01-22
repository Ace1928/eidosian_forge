from typing import AbstractSet, Any, Dict, Union
import numpy as np
import sympy
import cirq
from cirq import value, _compat
from cirq.ops import raw_types
@value.value_equality
class PhaseGradientGate(raw_types.Gate):
    """Phases all computational basis states proportional to the integer value of the state.

    The gate `cirq.PhaseGradientGate(n, t)` has the unitary
    $$
    \\sum_{x=0}^{2^n-1} \\omega^x |x\\rangle \\langle x|
    $$
    where
    $$
    \\omega=e^{2 \\pi i/2^n}
    $$

    This gate makes up a portion of the quantum fourier transform.
    """

    def __init__(self, *, num_qubits: int, exponent: Union[float, sympy.Basic]):
        self._num_qubits = num_qubits
        self._exponent = exponent

    @property
    def exponent(self) -> Union[float, sympy.Basic]:
        return self._exponent

    def _json_dict_(self) -> Dict[str, Any]:
        return {'num_qubits': self._num_qubits, 'exponent': self.exponent}

    def _value_equality_values_(self):
        return (self._num_qubits, self.exponent)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        for i, q in enumerate(qubits):
            yield (cirq.Z(q) ** (self.exponent / 2 ** i))

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs'):
        if isinstance(self.exponent, sympy.Basic):
            return NotImplemented
        n = int(np.prod([args.target_tensor.shape[k] for k in args.axes], dtype=np.int64))
        for i in range(n):
            p = 1j ** (4 * i / n * self.exponent)
            args.target_tensor[args.subspace_index(big_endian_bits_int=i)] *= p
        return args.target_tensor

    def __pow__(self, power):
        new_exponent = cirq.mul(self.exponent, power, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return PhaseGradientGate(num_qubits=self._num_qubits, exponent=new_exponent)

    def _unitary_(self):
        if isinstance(self.exponent, sympy.Basic):
            return NotImplemented
        size = 1 << self._num_qubits
        return np.diag([1j ** (4 * i / size * self.exponent) for i in range(size)])

    def _has_unitary_(self) -> bool:
        return not cirq.is_parameterized(self)

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.exponent)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.exponent)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'PhaseGradientGate':
        new_exponent = cirq.resolve_parameters(self.exponent, resolver, recursive)
        if new_exponent is self.exponent:
            return self
        return PhaseGradientGate(num_qubits=self._num_qubits, exponent=new_exponent)

    def __str__(self) -> str:
        return f'Grad[{self._num_qubits}]' + (f'^{self.exponent}' if self.exponent != 1 else '')

    def __repr__(self) -> str:
        return f'cirq.PhaseGradientGate(num_qubits={self._num_qubits!r}, exponent={_compat.proper_repr(self.exponent)})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(wire_symbols=('Grad',) + tuple((f'#{k + 1}' for k in range(1, self._num_qubits))), exponent=self.exponent, exponent_qubit_index=0)