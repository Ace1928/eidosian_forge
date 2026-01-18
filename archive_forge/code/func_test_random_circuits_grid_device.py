import networkx as nx
import pytest
import cirq
@pytest.mark.parametrize('qubits, n_moments, op_density, random_state', [(5 * size, 20 * size, density, seed) for size in range(1, 3) for seed in range(3) for density in [0.4, 0.5, 0.6]])
def test_random_circuits_grid_device(qubits: int, n_moments: int, op_density: float, random_state: int):
    c_orig = cirq.testing.random_circuit(qubits=qubits, n_moments=n_moments, op_density=op_density, random_state=random_state)
    mapping = glob_mapper.initial_mapping(c_orig)
    assert len(set(mapping.values())) == len(mapping.values())
    assert set(mapping.keys()) == set(c_orig.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(glob_device_graph, mapping.values()))