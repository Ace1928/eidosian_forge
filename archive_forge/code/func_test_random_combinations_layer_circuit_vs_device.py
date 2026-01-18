import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_random_combinations_layer_circuit_vs_device():
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3))
    layer_circuit = get_grid_interaction_layer_circuit(graph)
    combs1 = get_random_combinations_for_layer_circuit(n_library_circuits=10, n_combinations=10, layer_circuit=layer_circuit, random_state=1)
    combs2 = get_random_combinations_for_device(n_library_circuits=10, n_combinations=10, device_graph=graph, random_state=1)
    for comb1, comb2 in zip(combs1, combs2):
        assert comb1.pairs == comb2.pairs
        assert np.all(comb1.combinations == comb2.combinations)