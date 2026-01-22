import numpy as np
import scipy.stats
import cirq
def _test_decompose(matrix, controls_count):
    qubits = cirq.LineQubit.range(controls_count + 1)
    operations = cirq.decompose_multi_controlled_rotation(matrix, qubits[:-1], qubits[-1])
    _count_operations(operations)
    result_matrix = cirq.Circuit(operations).unitary()
    expected_matrix = cirq.Circuit([cirq.MatrixGate(matrix).on(qubits[-1]).controlled_by(*qubits[:-1])]).unitary()
    cirq.testing.assert_allclose_up_to_global_phase(expected_matrix, result_matrix, atol=1e-08)