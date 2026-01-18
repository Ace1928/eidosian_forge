import itertools
from typing import Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('depolarization, estimator', itertools.product((0.0, 0.2, 0.7, 1.0), (cirq.hog_score_xeb_fidelity_from_probabilities, cirq.linear_xeb_fidelity_from_probabilities, cirq.log_xeb_fidelity_from_probabilities)))
def test_xeb_fidelity(depolarization, estimator):
    prng_state = np.random.get_state()
    np.random.seed(0)
    fs = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(circuit, qubits, depolarization, repetitions=5000)
        f = cirq.xeb_fidelity(circuit, bitstrings, qubits, estimator=estimator)
        amplitudes = cirq.final_state_vector(circuit)
        f2 = cirq.xeb_fidelity(circuit, bitstrings, qubits, amplitudes=amplitudes, estimator=estimator)
        assert np.abs(f - f2) < 2e-06
        fs.append(f)
    estimated_fidelity = np.mean(fs)
    expected_fidelity = 1 - depolarization
    assert np.isclose(estimated_fidelity, expected_fidelity, atol=0.04)
    np.random.set_state(prng_state)