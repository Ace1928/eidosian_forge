from typing import Optional, List, Iterator, Iterable, TYPE_CHECKING
from cirq.interop.quirk.cells.cell import Cell, CELL_SIZES, CellMaker
class InputCell(Cell):
    """A modifier that provides a quantum input to gates in the same column."""

    def __init__(self, qubits: Iterable['cirq.Qid'], letter: str):
        self.qubits = tuple(qubits)
        self.letter = letter

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return InputCell(qubits=Cell._replace_qubits(self.qubits, qubits), letter=self.letter)

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            cell = column[i]
            if cell is not None:
                column[i] = cell.with_input(self.letter, self.qubits)