import abc
import dataclasses
from typing import Iterable, List, Optional
import cirq
from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs
class SymbolResolver(metaclass=abc.ABCMeta):
    """Abstract class providing the interface for users to specify information
    about how a particular symbol should be displayed in the 3D circuit
    """

    def __call__(self, operation: cirq.Operation) -> Optional[SymbolInfo]:
        return self.resolve(operation)

    @abc.abstractmethod
    def resolve(self, operation: cirq.Operation) -> Optional[SymbolInfo]:
        """Converts cirq.Operation objects into SymbolInfo objects for serialization."""