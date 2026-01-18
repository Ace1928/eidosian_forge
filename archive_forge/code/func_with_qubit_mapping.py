import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def with_qubit_mapping(self, qubit_map: Union[Mapping['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]) -> 'cirq.CircuitOperation':
    """Returns a copy of this operation with an updated qubit mapping.

        Users should pass either 'qubit_map' or 'transform' to this method.

        Args:
            qubit_map: A mapping of old qubits to new qubits. This map will be
                composed with any existing qubit mapping.

        Returns:
            A copy of this operation targeting qubits as indicated by qubit_map.

        Raises:
            TypeError: qubit_map was not a function or dict mapping qubits to
                qubits.
            ValueError: The new operation has a different number of qubits than
                this operation.
        """
    if callable(qubit_map):
        transform = qubit_map
    elif isinstance(qubit_map, dict):
        transform = lambda q: qubit_map.get(q, q)
    else:
        raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')
    new_map = {}
    for q in self.circuit.all_qubits():
        q_new = transform(self.qubit_map.get(q, q))
        if q_new != q:
            if q_new.dimension != q.dimension:
                raise ValueError(f'Qid dimension conflict.\nFrom qid: {q}\nTo qid: {q_new}')
            new_map[q] = q_new
    new_op = self.replace(qubit_map=new_map)
    if len(set(new_op.qubits)) != len(set(self.qubits)):
        raise ValueError(f'Collision in qubit map composition. Original map:\n{self.qubit_map}\nMap after changes: {new_op.qubit_map}')
    return new_op