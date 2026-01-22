import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
class UnknownOperation(cirq.Operation):

    def __init__(self, qubits):
        self._qubits = qubits

    @property
    def qubits(self):
        return self._qubits

    def with_qubits(self, *new_qubits):
        return UnknownOperation(self._qubits)