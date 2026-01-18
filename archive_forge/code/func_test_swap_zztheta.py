import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_swap_zztheta():
    qubits = cirq.LineQubit.range(2)
    a, b = qubits
    for theta in np.linspace(0, 2 * np.pi, 10):
        circuit = cirq.Circuit(cirq.SWAP(a, b), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(a, b))
        converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)