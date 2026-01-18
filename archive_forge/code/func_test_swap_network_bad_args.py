import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
def test_swap_network_bad_args():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    with pytest.raises(ValueError):
        initial_mapping = dict(zip(qubits, range(n_qubits)))
        ccr.SwapNetwork(circuit, initial_mapping)
    with pytest.raises(ValueError):
        initial_mapping = dict(zip(range(n_qubits), qubits))
        ccr.SwapNetwork(circuit, initial_mapping)