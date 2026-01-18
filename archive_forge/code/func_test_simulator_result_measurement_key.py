import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_measurement_key():
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100)
    assert result.probabilities() == {0: 0.2, 1: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100)
    assert result.probabilities('a') == {0: 1.0}
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [1]}, repetitions=100)
    assert result.probabilities('a') == {0: 0.2, 1: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [0, 1]}, repetitions=100)
    assert result.probabilities('a') == {0: 0.2, 1: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [1, 0]}, repetitions=100)
    assert result.probabilities('a') == {0: 0.2, 2: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 7: 0.8}, num_qubits=3, measurement_dict={'a': [2]}, repetitions=100)
    assert result.probabilities('a') == {0: 0.2, 1: 0.8}
    result = ionq.SimulatorResult({0: 0.2, 4: 0.8}, num_qubits=3, measurement_dict={'a': [1, 2]}, repetitions=100)
    assert result.probabilities('a') == {0: 1.0}