import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_unconstrained_undirected_graph_device_edge_eq():
    e = ccgdgd._UnconstrainedUndirectedGraphDeviceEdge()
    f = ccgd.UnconstrainedUndirectedGraphDeviceEdge
    assert e == f
    assert e != 3