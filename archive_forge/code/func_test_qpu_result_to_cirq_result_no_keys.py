import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_qpu_result_to_cirq_result_no_keys():
    result = ionq.QPUResult({0: 1, 1: 2}, num_qubits=2, measurement_dict={})
    with pytest.raises(ValueError, match='cirq results'):
        _ = result.to_cirq_result()