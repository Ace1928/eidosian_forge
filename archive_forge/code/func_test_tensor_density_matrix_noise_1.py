import numpy as np
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_density_matrix_noise_1():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]), cirq.amplitude_damp(0.01).on(q[0]), cirq.phase_damp(0.001).on(q[0]))
    rho1 = cirq.final_density_matrix(c, qubit_order=q, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(c, q)
    np.testing.assert_allclose(rho1, rho2, atol=1e-15)