import abc
import dataclasses
from typing import Iterable, List, Optional
import cirq
from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs
@staticmethod
def unknown_operation(num_qubits: int) -> 'SymbolInfo':
    """Generates a SymbolInfo object for an unknown operation.

        Args:
            num_qubits: the number of qubits in the operation
        """
    symbol_info = SymbolInfo([], [])
    for _ in range(num_qubits):
        symbol_info.colors.append('gray')
        symbol_info.labels.append('?')
    return symbol_info