from typing import DefaultDict, Dict, Sequence, TYPE_CHECKING, Optional
import abc
from collections import defaultdict
from cirq import circuits, devices, ops, protocols, transformers
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate
from cirq.contrib.acquaintance.permutation import (
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
class AcquaintanceOperation(ops.GateOperation):
    """Represents an a acquaintance opportunity between a particular set of
    logical indices on a particular set of physical qubits.
    """

    def __init__(self, qubits: Sequence['cirq.Qid'], logical_indices: Sequence[LogicalIndex]) -> None:
        if len(logical_indices) != len(qubits):
            raise ValueError('len(logical_indices) != len(qubits)')
        super().__init__(AcquaintanceOpportunityGate(num_qubits=len(qubits)), qubits)
        self.logical_indices: LogicalIndexSequence = logical_indices

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        wire_symbols = tuple((f'({i})' for i in self.logical_indices))
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)