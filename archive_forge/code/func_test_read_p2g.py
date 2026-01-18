import io
import networkx as nx
from networkx.readwrite.p2g import read_p2g, write_p2g
from networkx.utils import edges_equal
def test_read_p2g(self):
    s = b'name\n3 4\na\n1 2\nb\n\nc\n0 2\n'
    bytesIO = io.BytesIO(s)
    G = read_p2g(bytesIO)
    assert G.name == 'name'
    assert sorted(G) == ['a', 'b', 'c']
    edges = [(str(u), str(v)) for u, v in G.edges()]
    assert edges_equal(G.edges(), [('a', 'c'), ('a', 'b'), ('c', 'a'), ('c', 'c')])