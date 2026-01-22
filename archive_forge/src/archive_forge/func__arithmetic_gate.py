import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _arithmetic_gate(identifier: str, size: int, func: _IntsToIntCallable) -> CellMaker:
    operation = _QuirkArithmeticCallable(func)
    assert identifier not in ARITHMETIC_OP_TABLE
    ARITHMETIC_OP_TABLE[identifier] = operation
    return CellMaker(identifier=identifier, size=size, maker=lambda args: ArithmeticCell(identifier=identifier, target=args.qubits, inputs=[None] * len(operation.letters)))