import io
import networkx as nx
from networkx.readwrite.p2g import read_p2g, write_p2g
from networkx.utils import edges_equal
def test_write_p2g(self):
    s = b'foo\n3 2\n1\n1 \n2\n2 \n3\n\n'
    fh = io.BytesIO()
    G = nx.DiGraph()
    G.name = 'foo'
    G.add_edges_from([(1, 2), (2, 3)])
    write_p2g(G, fh)
    fh.seek(0)
    r = fh.read()
    assert r == s