import itertools
import random
import numpy as np
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('n_moments,algo,seed,make_bad', [(8, algo, random_seed(), make_bad) for algo in ccr.ROUTERS for make_bad in (False, True) for _ in range(5)] + [(0, 'greedy', random_seed(), False)])
def test_route_circuit_via_unitaries(n_moments, algo, seed, make_bad):
    circuit = cirq.testing.random_circuit(4, n_moments, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(3, 2)
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=seed)
    logical_qubits = sorted(circuit.all_qubits())
    if len(logical_qubits) < 2:
        return
    reverse_mapping = {l: p for p, l in swap_network.initial_mapping.items()}
    physical_qubits = [reverse_mapping[l] for l in logical_qubits]
    physical_qubits += list(set(device_graph).difference(physical_qubits))
    n_unused_qubits = len(physical_qubits) - len(logical_qubits)
    if make_bad:
        swap_network.circuit += [cirq.CNOT(*physical_qubits[:2])]
    cca.return_to_initial_mapping(swap_network.circuit)
    logical_unitary = circuit.unitary(qubit_order=logical_qubits)
    logical_unitary = np.kron(logical_unitary, np.eye(1 << n_unused_qubits))
    physical_unitary = swap_network.circuit.unitary(qubit_order=physical_qubits)
    assert ccr.is_valid_routing(circuit, swap_network) == (not make_bad)
    assert np.allclose(physical_unitary, logical_unitary) == (not make_bad)