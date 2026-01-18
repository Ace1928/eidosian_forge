from __future__ import annotations
from typing import Optional, Union, Iterable, TYPE_CHECKING
import itertools
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError
from .builder import ControlFlowBuilderBlock, InstructionPlaceholder, InstructionResources
from .control_flow import ControlFlowOp
from ._builder_utils import (
def with_false_block(self, false_block: ControlFlowBuilderBlock) -> 'IfElsePlaceholder':
    """Return a new placeholder instruction, with the false block set to the given value,
        updating the bits used by both it and the true body, if necessary.

        It is an error to try and set the false block on a placeholder that already has one.

        Args:
            false_block: The (unbuilt) instruction scope to set the false body to.

        Returns:
            A new placeholder, with ``false_block`` set to the given input, and both true and false
            blocks expanded to account for all resources.

        Raises:
            CircuitError: if the false block of this placeholder instruction is already set.
        """
    if self.__false_block is not None:
        raise CircuitError(f'false block is already set to {self.__false_block}')
    true_block = self.__true_block.copy()
    true_bits = true_block.qubits() | true_block.clbits()
    false_bits = false_block.qubits() | false_block.clbits()
    true_block.add_bits(false_bits - true_bits)
    false_block.add_bits(true_bits - false_bits)
    return type(self)(self.condition, true_block, false_block, label=self.label)