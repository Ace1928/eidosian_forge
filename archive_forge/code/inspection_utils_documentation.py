from typing import FrozenSet, Sequence, Set, TYPE_CHECKING
from cirq import devices
from cirq.contrib.acquaintance.executor import AcquaintanceOperation, ExecutionStrategy
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.permutation import LogicalIndex, LogicalMapping
from cirq.contrib import circuitdag
Inits LogicalAnnotator.

        Args:
            initial_mapping: The initial mapping of qubits to logical indices.
        