from typing import Optional
from qiskit.circuit.instruction import Instruction
from .builder import InstructionPlaceholder, InstructionResources
class BreakLoopOp(Instruction):
    """A circuit operation which, when encountered, jumps to the end of
    the nearest enclosing loop.

    .. note:

        Can be inserted only within the body of a loop op, and must span
        the full width of that block.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────────┐
        q_0: ┤0             ├
             │              │
        q_1: ┤1             ├
             │  break_loop  │
        q_2: ┤2             ├
             │              │
        c_0: ╡0             ╞
             └──────────────┘

    """

    def __init__(self, num_qubits: int, num_clbits: int, label: Optional[str]=None):
        super().__init__('break_loop', num_qubits, num_clbits, [], label=label)