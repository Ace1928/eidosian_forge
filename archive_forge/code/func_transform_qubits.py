import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
def transform_qubits(self, qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]) -> Self:
    """Returns the same operation, but with different qubits.

        Args:
            qubit_map: A function or a dict mapping each current qubit into a desired
                new qubit.

        Returns:
            The receiving operation but with qubits transformed by the given
                function.
        Raises:
            TypeError: qubit_map was not a function or dict mapping qubits to
                qubits.
        """
    if callable(qubit_map):
        transform = qubit_map
    elif isinstance(qubit_map, dict):
        transform = lambda q: qubit_map.get(q, q)
    else:
        raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')
    return self.with_qubits(*(transform(q) for q in self.qubits))