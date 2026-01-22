from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class DecomposableNoUnitary(cirq.Operation):
    qubits = ()
    with_qubits = NotImplemented

    def __init__(self, qubits):
        self.qubits = qubits

    def _decompose_(self):
        for q in self.qubits:
            yield ReturnsNotImplemented()(q)