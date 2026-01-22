from typing import Any, Dict, Sequence, Union
import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np
@cirq.value.value_equality
class GPIGate(cirq.Gate):
    """The GPI gate is a single qubit gate representing a pi pulse.

    The unitary matrix of this gate is:
    $$
    \\begin{bmatrix}
      0 & e^{-i 2\\pi\\phi} \\\\
      e^{-i2\\pi\\phi} & 0
    \\end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = cmath.exp(-self.phi * 2 * math.pi * 1j)
        bot = cmath.exp(self.phi * 2 * math.pi * 1j)
        return np.array([[0, top], [bot, 0]])

    def __str__(self) -> str:
        return 'GPI'

    def _num_qubits_(self) -> int:
        return 1

    @property
    def phase(self) -> float:
        return self.phi

    def __repr__(self) -> str:
        return f'cirq_ionq.GPIGate(phi={self.phi!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI({self.phase!r})',))

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self
        return NotImplemented