from typing import (
import re
import warnings
from dataclasses import dataclass
import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops
def validate_operation(self, operation: cirq.Operation) -> None:
    """Raises an exception if an operation is not valid.

        An operation is valid if
            * The operation is in the device gateset.
            * The operation targets a valid qubit
            * The operation targets a valid qubit pair, if it is a two-qubit operation.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """
    if operation not in self._metadata.gateset:
        raise ValueError(f'Operation {operation} contains a gate which is not supported.')
    for q in operation.qubits:
        if q not in self._metadata.qubit_set:
            raise ValueError(f'Qubit not on device: {q!r}.')
    if len(operation.qubits) == 2 and (not any((operation in gf for gf in _VARIADIC_GATE_FAMILIES))) and (frozenset(operation.qubits) not in self._metadata.qubit_pairs):
        raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')