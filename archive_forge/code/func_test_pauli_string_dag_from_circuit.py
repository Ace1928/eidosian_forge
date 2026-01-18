import cirq
from cirq.contrib.paulistring import convert_and_separate_circuit, pauli_string_dag_from_circuit
def test_pauli_string_dag_from_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)
    c_left, _ = convert_and_separate_circuit(c_orig)
    c_left_dag = pauli_string_dag_from_circuit(c_left)
    c_left_reordered = c_left_dag.to_circuit()
    cirq.testing.assert_allclose_up_to_global_phase(c_left.unitary(), c_left_reordered.unitary(), atol=1e-07)