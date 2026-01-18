import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_to_cirq_result():
    result = ionq.SimulatorResult({0: 0.25, 1: 0.75}, num_qubits=2, measurement_dict={'x': [0, 1]}, repetitions=3)
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 0], [0, 1]])})
    assert result.to_cirq_result(seed=3) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 1], [0, 1]])})
    params = cirq.ParamResolver({'a': 0.1})
    assert result.to_cirq_result(seed=3, params=params) == cirq.ResultDict(params=params, measurements={'x': np.array([[0, 1], [0, 1], [0, 1]])})
    assert result.to_cirq_result(seed=2, override_repetitions=2) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 0]])})
    result = ionq.SimulatorResult({0: 0.25, 1: 0.75}, num_qubits=2, measurement_dict={'x': [0]}, repetitions=3)
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [0], [0]])})
    result = ionq.SimulatorResult({0: 0.25, 1: 0.75}, num_qubits=2, measurement_dict={'x': [1]}, repetitions=3)
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[1], [0], [1]])})
    assert type(result.to_cirq_result().measurements['x']) == np.ndarray