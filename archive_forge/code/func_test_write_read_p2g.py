import io
import networkx as nx
from networkx.readwrite.p2g import read_p2g, write_p2g
from networkx.utils import edges_equal
def test_write_read_p2g(self):
    fh = io.BytesIO()
    G = nx.DiGraph()
    G.name = 'foo'
    G.add_edges_from([('a', 'b'), ('b', 'c')])
    write_p2g(G, fh)
    fh.seek(0)
    H = read_p2g(fh)
    assert edges_equal(G.edges(), H.edges())