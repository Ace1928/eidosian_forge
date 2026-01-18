import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def with_operation(self, operation: 'cirq.Operation') -> 'cirq.Moment':
    """Returns an equal moment, but with the given op added.

        Args:
            operation: The operation to append.

        Returns:
            The new moment.

        Raises:
            ValueError: If the operation given overlaps a current operation in the moment.
        """
    if any((q in self._qubits for q in operation.qubits)):
        raise ValueError(f'Overlapping operations: {operation}')
    m = Moment(_flatten_contents=False)
    m._operations = self._operations + (operation,)
    m._sorted_operations = None
    m._qubits = self._qubits.union(operation.qubits)
    m._qubit_to_op = {**self._qubit_to_op, **{q: operation for q in operation.qubits}}
    m._measurement_key_objs = self._measurement_key_objs_().union(protocols.measurement_key_objs(operation))
    m._control_keys = self._control_keys_().union(protocols.control_keys(operation))
    return m