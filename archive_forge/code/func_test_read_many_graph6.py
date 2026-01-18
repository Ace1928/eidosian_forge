import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_read_many_graph6(self):
    data = b':Q___eDcdFcDeFcE`GaJ`IaHbKNbLM\n:Q___dCfDEdcEgcbEGbFIaJ`JaHN`IM'
    fh = BytesIO(data)
    glist = nx.read_sparse6(fh)
    assert len(glist) == 2
    for G in glist:
        assert nodes_equal(G.nodes(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])