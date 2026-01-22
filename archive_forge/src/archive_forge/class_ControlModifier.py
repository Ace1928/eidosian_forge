from __future__ import annotations
import dataclasses
from typing import Union, List
from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError
@dataclasses.dataclass
class ControlModifier(Modifier):
    """Control modifier: specifies that the operation is controlled by ``num_ctrl_qubits``
    and has control state ``ctrl_state``."""
    num_ctrl_qubits: int = 0
    ctrl_state: Union[int, str, None] = None

    def __init__(self, num_ctrl_qubits: int=0, ctrl_state: Union[int, str, None]=None):
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)