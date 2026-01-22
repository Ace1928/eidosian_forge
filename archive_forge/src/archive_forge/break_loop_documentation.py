from typing import Optional
from qiskit.circuit.instruction import Instruction
from .builder import InstructionPlaceholder, InstructionResources
A placeholder instruction for use in control-flow context managers, when the number of qubits
    and clbits is not yet known.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    