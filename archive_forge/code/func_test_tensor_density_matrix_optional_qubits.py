import numpy as np
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_density_matrix_optional_qubits():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))
    rho1 = cirq.final_density_matrix(c, dtype=np.complex128)
    rho2 = ccq.tensor_density_matrix(c)
    np.testing.assert_allclose(rho1, rho2, atol=1e-15)