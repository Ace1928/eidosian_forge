import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_simulator_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100), ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.4, 1: 0.6}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'b': [0]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [1]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0, 1]}, repetitions=100))
    equals_tester.add_equality_group(ionq.SimulatorResult({0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0, 1]}, repetitions=10))