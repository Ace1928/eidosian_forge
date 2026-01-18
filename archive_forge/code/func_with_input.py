from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
def with_input(self, letter: str, register: Union[Sequence['cirq.Qid'], int]) -> 'CompositeCell':
    return self._transform_cells(lambda cell: cell.with_input(letter, register))