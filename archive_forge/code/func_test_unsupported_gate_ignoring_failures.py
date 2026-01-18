import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_unsupported_gate_ignoring_failures():

    class UnknownOperation(cirq.Operation):

        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            return UnknownOperation(self._qubits)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(UnknownOperation([q0]))
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    assert circuit == converted_circuit