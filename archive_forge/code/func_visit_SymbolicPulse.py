from the parameter table model of ~O(1), however, usually, this calculation occurs
from each object, yielding smaller object creation cost and higher performance
from __future__ import annotations
from copy import copy
from typing import Any
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import instructions, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import SymbolicPulse, Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.pulse.utils import format_parameter_value
def visit_SymbolicPulse(self, node: SymbolicPulse):
    """Get parameters from ``SymbolicPulse`` object."""
    for op_value in node.parameters.values():
        if isinstance(op_value, ParameterExpression):
            self.parameters |= op_value.parameters