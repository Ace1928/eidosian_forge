import itertools
import random
import numpy as np
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('n_moments,algo,circuit_seed,routing_seed', [(20, algo, random_seed(), random_seed()) for algo in ccr.ROUTERS for _ in range(5)] + [(0, 'greedy', random_seed(), random_seed())] + [(10, 'greedy', random_seed(), None)])
def test_route_circuit(n_moments, algo, circuit_seed, routing_seed):
    circuit = cirq.testing.random_circuit(10, n_moments, 0.5, random_state=circuit_seed)
    device_graph = ccr.get_grid_device_graph(4, 3)
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=routing_seed)
    assert set(swap_network.initial_mapping).issubset(device_graph)
    assert sorted(swap_network.initial_mapping.values()) == sorted(circuit.all_qubits())
    assert ccr.ops_are_consistent_with_device_graph(swap_network.circuit.all_operations(), device_graph)
    assert ccr.is_valid_routing(circuit, swap_network)