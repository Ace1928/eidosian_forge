import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_numpy_float64_inference(self):
    np = pytest.importorskip('numpy')
    G = self.attribute_numeric_type_graph
    G.edges['n1', 'n1']['weight'] = np.float64(1.1)
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname, infer_numeric_types=True)
    H = nx.read_graphml(fname)
    assert G._adj == H._adj
    os.close(fd)
    os.unlink(fname)