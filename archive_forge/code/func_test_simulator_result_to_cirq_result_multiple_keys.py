import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_to_cirq_result_multiple_keys():
    result = ionq.SimulatorResult({0: 0.25, 3: 0.75}, num_qubits=3, measurement_dict={'x': [1], 'y': [2, 0]}, repetitions=3)
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'x': np.array([[1], [0], [1]]), 'y': np.array([[1, 0], [0, 0], [1, 0]])})