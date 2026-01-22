from typing import Iterable
import cirq
from cirq_web import widget
from cirq_web.circuits.symbols import (
def _build_3D_symbol(self, operation, moment) -> Operation3DSymbol:
    symbol_info = resolve_operation(operation, self._resolvers)
    location_info = []
    for qubit in operation.qubits:
        if isinstance(qubit, cirq.GridQubit):
            location_info.append({'row': qubit.row, 'col': qubit.col})
        elif isinstance(qubit, cirq.LineQubit):
            location_info.append({'row': qubit.x, 'col': 0})
        else:
            raise ValueError('Unsupported qubit type')
    return Operation3DSymbol(symbol_info.labels, location_info, symbol_info.colors, moment)