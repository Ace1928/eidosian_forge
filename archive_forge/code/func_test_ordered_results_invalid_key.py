import pytest
import numpy as np
import cirq_ionq as ionq
import cirq.testing
def test_ordered_results_invalid_key():
    result = ionq.QPUResult({0: 1, 1: 2}, num_qubits=2, measurement_dict={'x': [1]})
    with pytest.raises(ValueError, match='is not a key for'):
        _ = result.ordered_results('y')