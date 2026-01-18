import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_unconstrained_undirected_graph_device_edge():
    edge = ccgd.UnconstrainedUndirectedGraphDeviceEdge
    qubits = cirq.LineQubit.range(2)
    assert edge.duration_of(cirq.X(qubits[0])) == cirq.Duration(picos=0)
    assert edge.duration_of(cirq.CZ(*qubits[:2])) == cirq.Duration(picos=0)