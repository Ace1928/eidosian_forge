import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_measurement_key():
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [0]})
    assert result.counts() == {0: 10, 1: 20}
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [0]})
    assert result.counts('a') == {0: 30}
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [1]})
    assert result.counts('a') == {0: 10, 1: 20}
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [0, 1]})
    assert result.counts('a') == {0: 10, 1: 20}
    result = ionq.QPUResult({0: 10, 1: 20}, num_qubits=2, measurement_dict={'a': [1, 0]})
    assert result.counts('a') == {0: 10, 2: 20}
    result = ionq.QPUResult({0: 10, 7: 20}, num_qubits=3, measurement_dict={'a': [2]})
    assert result.counts('a') == {0: 10, 1: 20}
    result = ionq.QPUResult({0: 10, 4: 20}, num_qubits=3, measurement_dict={'a': [1, 2]})
    assert result.counts('a') == {0: 30}