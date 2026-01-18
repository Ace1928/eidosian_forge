import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_str():
    result = ionq.SimulatorResult({0: 0.4, 1: 0.6}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100)
    assert str(result) == '00: 0.4\n01: 0.6'