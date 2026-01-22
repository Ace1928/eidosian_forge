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
def _assign_parameter_expression(self, param_expr: ParameterExpression):
    """A helper function to assign parameter value to parameter expression."""
    new_value = copy(param_expr)
    updated = param_expr.parameters & self._param_map.keys()
    for param in updated:
        new_value = new_value.assign(param, self._param_map[param])
    new_value = format_parameter_value(new_value)
    return new_value