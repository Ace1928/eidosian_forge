import cirq
from cirq.contrib.paulistring import clifford_optimized_circuit, CliffordTargetGateset
def test_optimize_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)
    c_opt = clifford_optimized_circuit(c_orig)
    cirq.testing.assert_allclose_up_to_global_phase(c_orig.unitary(), c_opt.unitary(), atol=1e-07)