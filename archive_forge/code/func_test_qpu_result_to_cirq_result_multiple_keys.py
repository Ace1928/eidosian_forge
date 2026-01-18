import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_to_cirq_result_multiple_keys():
    result = ionq.QPUResult({0: 2, 5: 3}, num_qubits=3, measurement_dict={'x': [1], 'y': [2, 0]})
    assert result.to_cirq_result() == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [0], [0], [0], [0]]), 'y': np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])})