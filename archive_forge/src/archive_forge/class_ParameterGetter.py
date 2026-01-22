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
class ParameterGetter(NodeVisitor):
    """Node visitor for parameter finding.

    This visitor initializes empty parameter array, and recursively visits nodes
    and add parameters found to the array.
    """

    def __init__(self):
        self.parameters = set()

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        """Visit ``ScheduleBlock``. Recursively visit context blocks and search parameters.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
        self.parameters |= node._parameter_manager.parameters

    def visit_Schedule(self, node: Schedule):
        """Visit ``Schedule``. Recursively visit schedule children and search parameters."""
        self.parameters |= node.parameters

    def visit_AlignmentKind(self, node: AlignmentKind):
        """Get parameters from block's ``AlignmentKind`` specification."""
        for param in node._context_params:
            if isinstance(param, ParameterExpression):
                self.parameters |= param.parameters

    def visit_Instruction(self, node: instructions.Instruction):
        """Get parameters from general pulse instruction.

        .. note:: All parametrized object should be stored in the operands.
            Otherwise, parameter cannot be detected.
        """
        for op in node.operands:
            self.visit(op)

    def visit_Channel(self, node: channels.Channel):
        """Get parameters from ``Channel`` object."""
        self.parameters |= node.parameters

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        """Get parameters from ``SymbolicPulse`` object."""
        for op_value in node.parameters.values():
            if isinstance(op_value, ParameterExpression):
                self.parameters |= op_value.parameters

    def visit_Waveform(self, node: Waveform):
        """Get parameters from ``Waveform`` object.

        .. node:: No parameter can be assigned to ``Waveform`` object.
        """
        pass

    def generic_visit(self, node: Any):
        """Get parameters from object that doesn't belong to Qiskit Pulse module."""
        if isinstance(node, ParameterExpression):
            self.parameters |= node.parameters