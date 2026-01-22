from typing import Iterator, Optional, cast, Iterable, TYPE_CHECKING
from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker, ExplicitOperationsCell
def _measurement(identifier: str, basis_change: Optional['cirq.Gate']=None) -> CellMaker:
    return CellMaker(identifier=identifier, size=1, maker=lambda args: ExplicitOperationsCell([ops.measure(*args.qubits, key=f'row={args.row},col={args.col}')], basis_change=cast(Iterable['cirq.Operation'], [basis_change.on(*args.qubits)] if basis_change else ())))