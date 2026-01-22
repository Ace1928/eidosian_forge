from typing import Optional
from qiskit.circuit.instruction import Instruction
from .builder import InstructionPlaceholder, InstructionResources
class BreakLoopPlaceholder(InstructionPlaceholder):
    """A placeholder instruction for use in control-flow context managers, when the number of qubits
    and clbits is not yet known.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, *, label: Optional[str]=None):
        super().__init__('break_loop', 0, 0, [], label=label)

    def concrete_instruction(self, qubits, clbits):
        return (self._copy_mutable_properties(BreakLoopOp(len(qubits), len(clbits), label=self.label)), InstructionResources(qubits=tuple(qubits), clbits=tuple(clbits)))

    def placeholder_resources(self):
        return InstructionResources()