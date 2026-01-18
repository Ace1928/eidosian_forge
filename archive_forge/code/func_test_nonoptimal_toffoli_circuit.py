import cirq
def test_nonoptimal_toffoli_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2).unitary(), cirq.unitary(cirq.TOFFOLI(q0, q1, q2)), atol=1e-07)