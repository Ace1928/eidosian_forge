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
def update_parameter_table(self, new_node: Any):
    """A helper function to update parameter table with given data node.

        Args:
            new_node: A new data node to be added.
        """
    visitor = ParameterGetter()
    visitor.visit(new_node)
    self._parameters |= visitor.parameters