import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_nx_qubit_layout_3():
    g = nx.from_edgelist([(cirq.NamedQubit('a'), cirq.NamedQubit('b')), (cirq.NamedQubit('b'), cirq.NamedQubit('c'))])
    node_to_i = {cirq.NamedQubit('a'): 0, cirq.NamedQubit('b'): 1, cirq.NamedQubit('c'): 2}
    pos = ccr.nx_qubit_layout(g)
    for k, (x, y) in pos.items():
        assert x == 0.5
        assert y == node_to_i[k] + 1