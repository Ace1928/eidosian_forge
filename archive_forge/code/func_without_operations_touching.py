import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def without_operations_touching(self, qubits: Iterable['cirq.Qid']) -> 'cirq.Moment':
    """Returns an equal moment, but without ops on the given qubits.

        Args:
            qubits: Operations that touch these will be removed.

        Returns:
            The new moment.
        """
    qubits = frozenset(qubits)
    if not self.operates_on(qubits):
        return self
    return Moment((operation for operation in self.operations if qubits.isdisjoint(frozenset(operation.qubits))))