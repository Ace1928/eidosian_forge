import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_numpy_float64(self):
    np = pytest.importorskip('numpy')
    wt = np.float64(3.4)
    G = nx.Graph([(1, 2, {'weight': wt})])
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname)
    H = nx.read_graphml(fname, node_type=int)
    assert G.edges == H.edges
    wtG = G[1][2]['weight']
    wtH = H[1][2]['weight']
    assert wtG == pytest.approx(wtH, abs=1e-06)
    assert type(wtG) == np.float64
    assert type(wtH) == float
    os.close(fd)
    os.unlink(fname)