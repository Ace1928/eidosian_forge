import cirq
import pytest
@pytest.mark.parametrize('n_qubits, n_moments, op_density, seed', [(8, size, op_density, seed) for size in [50, 100] for seed in range(3) for op_density in [0.3, 0.5, 0.7]])
def test_route_small_circuit_random(n_qubits, n_moments, op_density, seed):
    c_orig = cirq.testing.random_circuit(qubits=n_qubits, n_moments=n_moments, op_density=op_density, random_state=seed)
    device = cirq.testing.construct_grid_device(4, 4)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed, imap, swap_map = router.route_circuit(c_orig, tag_inserted_swaps=True)
    device.validate_circuit(c_routed)
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(c_routed, c_orig.transform_qubits(imap), swap_map)