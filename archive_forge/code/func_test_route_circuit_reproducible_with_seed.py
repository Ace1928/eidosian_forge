import itertools
import random
import numpy as np
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('algo,seed', [(algo, random_seed()) for algo in ccr.ROUTERS for _ in range(3)])
def test_route_circuit_reproducible_with_seed(algo, seed):
    circuit = cirq.testing.random_circuit(8, 20, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(4, 3)
    wrappers = (lambda s: s, np.random.RandomState)
    swap_networks = []
    for wrapper, _ in itertools.product(wrappers, range(3)):
        swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=wrapper(seed))
        swap_networks.append(swap_network)
    eq = cirq.testing.equals_tester.EqualsTester()
    eq.add_equality_group(*swap_networks)