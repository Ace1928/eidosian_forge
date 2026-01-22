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
@property
def appended_instructions(self) -> Union[InstructionSet, None]:
    """Get the instruction set that was created when this block finished.  If the block has not
        yet finished, then this will be ``None``."""
    return self._appended_instructions