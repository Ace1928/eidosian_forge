import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_to_cirq_result():
    result = ionq.QPUResult({0: 1, 1: 2}, num_qubits=2, measurement_dict={'x': [0, 1]})
    assert result.to_cirq_result() == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 0], [0, 1], [0, 1]])})
    params = cirq.ParamResolver({'a': 0.1})
    assert result.to_cirq_result(params) == cirq.ResultDict(params=params, measurements={'x': np.array([[0, 0], [0, 1], [0, 1]])})
    result = ionq.QPUResult({0: 1, 1: 2}, num_qubits=2, measurement_dict={'x': [0]})
    assert result.to_cirq_result() == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [0], [0]])})
    result = ionq.QPUResult({0: 1, 1: 2}, num_qubits=2, measurement_dict={'x': [1]})
    assert result.to_cirq_result() == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [1], [1]])})
    assert type(result.to_cirq_result().measurements['x']) == np.ndarray
    result = ionq.QPUResult({2: 1, 1: 2}, num_qubits=2, measurement_dict={'x': [0, 1], 'y': [0], 'z': [1]})
    assert result.to_cirq_result() == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 1], [1, 0]]), 'y': np.array([[0], [0], [1]]), 'z': np.array([[1], [1], [0]])})