import cirq
from cirq.contrib.paulistring import optimized_circuit
def test_repeat_limit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)
    c_opt = optimized_circuit(c_orig, repeat=1)
    cirq.testing.assert_allclose_up_to_global_phase(c_orig.unitary(), c_opt.unitary(), atol=1e-07)
    assert sum((1 for op in c_opt.all_operations() if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate))) >= 10