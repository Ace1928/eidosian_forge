from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
@value.value_equality
class ControlCell(Cell):
    """A modifier that adds controls to other cells in the column."""

    def __init__(self, qubit: 'cirq.Qid', basis_change: Iterable['cirq.Operation']):
        self.qubit = qubit
        self._basis_change = tuple(basis_change)

    def _value_equality_values_(self) -> Any:
        return (self.qubit, self._basis_change)

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.cells.control_cells.ControlCell(\n    {self.qubit!r},\n    {self._basis_change!r})'

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ControlCell(qubit=Cell._replace_qubit(self.qubit, qubits), basis_change=tuple((op.with_qubits(*Cell._replace_qubits(op.qubits, qubits)) for op in self._basis_change)))

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not None:
                column[i] = gate.controlled_by(self.qubit)

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change