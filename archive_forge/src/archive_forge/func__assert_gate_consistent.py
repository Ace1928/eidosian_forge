from typing import Sequence, Optional, Union, Collection
from cirq import devices, ops, protocols
import numpy as np
def _assert_gate_consistent(gate: ops.Gate, num_controls: int, control_values: Optional[Sequence[Union[int, Collection[int]]]]) -> None:
    gate_controlled = gate.controlled(num_controls, control_values)
    qubits = devices.LineQid.for_gate(gate_controlled)
    control_qubits = qubits[:num_controls]
    gate_qubits = qubits[num_controls:]
    gate_controlled_on = gate_controlled.on(*control_qubits, *gate_qubits)
    gate_on_controlled_by = gate.on(*gate_qubits).controlled_by(*control_qubits, control_values=control_values)
    assert gate_controlled_on == gate_on_controlled_by, 'gate.controlled().on() and gate.on().controlled() should return the same operations.'