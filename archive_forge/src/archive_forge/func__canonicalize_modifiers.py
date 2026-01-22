from __future__ import annotations
import dataclasses
from typing import Union, List
from qiskit.circuit.operation import Operation
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.circuit.exceptions import CircuitError
def _canonicalize_modifiers(modifiers):
    """
    Returns the canonical representative of the modifier list. This is possible
    since all the modifiers commute; also note that InverseModifier is a special
    case of PowerModifier. The current solution is to compute the total number
    of control qubits / control state and the total power. The InverseModifier
    will be present if total power is negative, whereas the power modifier will
    be present only with positive powers different from 1.
    """
    power = 1
    num_ctrl_qubits = 0
    ctrl_state = 0
    for modifier in modifiers:
        if isinstance(modifier, InverseModifier):
            power *= -1
        elif isinstance(modifier, ControlModifier):
            num_ctrl_qubits += modifier.num_ctrl_qubits
            ctrl_state = ctrl_state << modifier.num_ctrl_qubits | modifier.ctrl_state
        elif isinstance(modifier, PowerModifier):
            power *= modifier.power
        else:
            raise CircuitError(f'Unknown modifier {modifier}.')
    canonical_modifiers = []
    if power < 0:
        canonical_modifiers.append(InverseModifier())
        power *= -1
    if power != 1:
        canonical_modifiers.append(PowerModifier(power))
    if num_ctrl_qubits > 0:
        canonical_modifiers.append(ControlModifier(num_ctrl_qubits, ctrl_state))
    return canonical_modifiers