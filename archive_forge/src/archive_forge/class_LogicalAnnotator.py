from typing import FrozenSet, Sequence, Set, TYPE_CHECKING
from cirq import devices
from cirq.contrib.acquaintance.executor import AcquaintanceOperation, ExecutionStrategy
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.permutation import LogicalIndex, LogicalMapping
from cirq.contrib import circuitdag
class LogicalAnnotator(ExecutionStrategy):
    """Realizes acquaintance opportunities."""

    def __init__(self, initial_mapping: LogicalMapping) -> None:
        """Inits LogicalAnnotator.

        Args:
            initial_mapping: The initial mapping of qubits to logical indices.
        """
        self._initial_mapping = initial_mapping.copy()

    @property
    def initial_mapping(self) -> LogicalMapping:
        return self._initial_mapping

    @property
    def device(self) -> 'cirq.Device':
        return devices.UNCONSTRAINED_DEVICE

    def get_operations(self, indices: Sequence[LogicalIndex], qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        yield AcquaintanceOperation(qubits, indices)