import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_measurement_multiple_key():
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]})
    assert result.counts('a') == {0: 30}
    assert result.counts('b') == {0: 10, 1: 20}
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [1], 'b': [0]})
    assert result.counts('a') == {0: 10, 1: 20}
    assert result.counts('b') == {0: 30}