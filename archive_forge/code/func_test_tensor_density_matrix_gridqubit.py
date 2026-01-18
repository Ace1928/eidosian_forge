import numpy as np
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_density_matrix_gridqubit():
    qubits = cirq.GridQubit.rect(2, 2)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    circuit = cirq.drop_empty_moments(circuit)
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=0.001))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, qubits))
    rho1 = cirq.final_density_matrix(circuit, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(circuit, qubits)
    np.testing.assert_allclose(rho1, rho2, atol=1e-08)