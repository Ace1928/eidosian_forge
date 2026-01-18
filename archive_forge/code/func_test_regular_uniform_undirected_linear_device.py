import pytest
import cirq
import cirq.contrib.graph_device as ccgd
@pytest.mark.parametrize('arity', range(1, 5))
def test_regular_uniform_undirected_linear_device(arity):
    n_qubits = 10
    edge_labels = {arity: None}
    device = ccgd.uniform_undirected_linear_device(n_qubits, edge_labels)
    assert device.qubits == tuple(cirq.LineQubit.range(n_qubits))
    assert len(device.edges) == n_qubits - arity + 1
    for edge, label in device.labelled_edges.items():
        assert label is None
        assert len(edge) == arity