from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
def _sub_cell_cols_sealed(self) -> List[List[Optional[Cell]]]:
    if not isinstance(self._sub_cell_cols_generator, list):
        self._sub_cell_cols_generator = list(self._sub_cell_cols_generator)
    return cast(List[List[Optional[Cell]]], self._sub_cell_cols_generator)