from __future__ import annotations
import typing
from abc import ABC, abstractmethod
from qiskit.circuit.instruction import Instruction
class ControlFlowOp(Instruction, ABC):
    """Abstract class to encapsulate all control flow operations."""

    @property
    @abstractmethod
    def blocks(self) -> tuple[QuantumCircuit, ...]:
        """Tuple of QuantumCircuits which may be executed as part of the
        execution of this ControlFlowOp. May be parameterized by a loop
        parameter to be resolved at run time.
        """

    @abstractmethod
    def replace_blocks(self, blocks: typing.Iterable[QuantumCircuit]) -> ControlFlowOp:
        """Replace blocks and return new instruction.
        Args:
            blocks: Tuple of QuantumCircuits to replace in instruction.

        Returns:
            New ControlFlowOp with replaced blocks.
        """