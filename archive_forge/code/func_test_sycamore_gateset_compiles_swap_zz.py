import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_sycamore_gateset_compiles_swap_zz():
    qubits = cirq.LineQubit.range(3)
    gamma = np.random.randn()
    circuit1 = cirq.Circuit(cirq.SWAP(qubits[0], qubits[1]), cirq.Z(qubits[2]), cirq.ZZ(qubits[0], qubits[1]) ** gamma, strategy=cirq.InsertStrategy.NEW)
    circuit2 = cirq.Circuit(cirq.ZZ(qubits[0], qubits[1]) ** gamma, cirq.Z(qubits[2]), cirq.SWAP(qubits[0], qubits[1]), strategy=cirq.InsertStrategy.NEW)
    gateset = cirq_google.SycamoreTargetGateset()
    compiled_circuit1 = cirq.optimize_for_target_gateset(circuit1, gateset=gateset)
    compiled_circuit2 = cirq.optimize_for_target_gateset(circuit2, gateset=gateset)
    cirq.testing.assert_same_circuits(compiled_circuit1, compiled_circuit2)
    assert len(list(compiled_circuit1.findall_operations_with_gate_type(cirq_google.SycamoreGate))) == 3
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit1, compiled_circuit1, atol=1e-07)