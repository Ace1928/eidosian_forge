import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def with_qubits(self, *new_qubits):
    return UnknownOperation(self._qubits)