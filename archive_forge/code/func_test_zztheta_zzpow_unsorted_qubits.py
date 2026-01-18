import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_zztheta_zzpow_unsorted_qubits():
    qubits = (cirq.LineQubit(1), cirq.LineQubit(0))
    exponent = 0.06366197723675814
    circuit = cirq.Circuit(cirq.ZZPowGate(exponent=exponent, global_shift=-0.5).on(qubits[0], qubits[1]))
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)