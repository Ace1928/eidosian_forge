import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
def test_qsd_n_qubit_errors():
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(3)]
    with pytest.raises(ValueError, match='shaped numpy array'):
        cirq.Circuit(quantum_shannon_decomposition(qubits, np.eye(9)))
    with pytest.raises(ValueError, match='is_unitary'):
        cirq.Circuit(quantum_shannon_decomposition(qubits, np.ones((8, 8))))