from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
@value.value_equality(unhashable=True)
class SwapCell(Cell):

    def __init__(self, qubits: Iterable['cirq.Qid'], controls: Iterable['cirq.Qid']):
        self._qubits = list(qubits)
        self._controls = list(controls)

    def gate_count(self) -> int:
        return 1

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return SwapCell(qubits=Cell._replace_qubits(self._qubits, qubits), controls=Cell._replace_qubits(self._controls, qubits))

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is not self and isinstance(gate, SwapCell):
                assert self._controls == gate._controls
                self._qubits += gate._qubits
                column[i] = None

    def operations(self) -> 'cirq.OP_TREE':
        if len(self._qubits) != 2:
            raise ValueError('Wrong number of swap gates in a column.')
        return ops.SWAP(*self._qubits).controlled_by(*self._controls)

    def controlled_by(self, qubit: 'cirq.Qid'):
        return SwapCell(self._qubits, self._controls + [qubit])

    def _value_equality_values_(self) -> Any:
        return (self._qubits, self._controls)

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.cells.swap_cell.SwapCell(\n    {self._qubits!r},\n    {self._controls!r})'