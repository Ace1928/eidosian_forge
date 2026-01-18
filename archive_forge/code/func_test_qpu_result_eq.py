import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]}), ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 20}, num_qubits=1, measurement_dict={'a': [0]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 15, 1: 15}, num_qubits=1, measurement_dict={'a': [0]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 10}, num_qubits=2, measurement_dict={'a': [0]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'b': [0]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [1]}))
    equals_tester.add_equality_group(ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0, 1]}))