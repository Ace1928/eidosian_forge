import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_nx_qubit_layout_2():
    g = nx.from_edgelist([(cirq.LineQubit(0), cirq.LineQubit(1)), (cirq.LineQubit(1), cirq.LineQubit(2))])
    pos = ccr.nx_qubit_layout(g)
    for k, (x, y) in pos.items():
        assert x == k.x
        assert y == 0.5