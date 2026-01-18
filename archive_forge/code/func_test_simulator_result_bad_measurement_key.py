import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_bad_measurement_key():
    result = ionq.SimulatorResult({0: 0.2, 1: 0.8}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}, repetitions=100)
    with pytest.raises(ValueError, match='bad'):
        result.probabilities('bad')