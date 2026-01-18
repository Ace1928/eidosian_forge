import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('circuits', [[cirq.testing.random_circuit(10, 10, 0.5) for _ in range(3)]])
def test_swap_network_equality(circuits):
    et = cirq.testing.EqualsTester()
    for circuit in circuits:
        qubits = sorted(circuit.all_qubits())
        for y in (0, 1):
            mapping = {cirq.GridQubit(x, y): q for x, q in enumerate(qubits)}
            et.add_equality_group(ccr.SwapNetwork(circuit, mapping))