import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
class TestEdgelist:

    @classmethod
    def setup_class(cls):
        cls.G = nx.Graph(name='test')
        e = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('a', 'f')]
        cls.G.add_edges_from(e)
        cls.G.add_node('g')
        cls.DG = nx.DiGraph(cls.G)
        cls.XG = nx.MultiGraph()
        cls.XG.add_weighted_edges_from([(1, 2, 5), (1, 2, 5), (1, 2, 1), (3, 3, 42)])
        cls.XDG = nx.MultiDiGraph(cls.XG)

    def test_write_edgelist_1(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        nx.write_edgelist(G, fh, data=False)
        fh.seek(0)
        assert fh.read() == b'1 2\n2 3\n'

    def test_write_edgelist_2(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        nx.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b'1 2 {}\n2 3 {}\n'

    def test_write_edgelist_3(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        nx.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b"1 2 {'weight': 2.0}\n2 3 {'weight': 3.0}\n"

    def test_write_edgelist_4(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        nx.write_edgelist(G, fh, data=['weight'])
        fh.seek(0)
        assert fh.read() == b'1 2 2.0\n2 3 3.0\n'

    def test_unicode(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, 'Radiohead', **{name2: 3})
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname)
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_latin1_issue(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, 'Radiohead', **{name2: 3})
        fd, fname = tempfile.mkstemp()
        pytest.raises(UnicodeEncodeError, nx.write_edgelist, G, fname, encoding='latin-1')
        os.close(fd)
        os.unlink(fname)

    def test_latin1(self):
        G = nx.Graph()
        name1 = 'Bj' + chr(246) + 'rk'
        name2 = chr(220) + 'ber'
        G.add_edge(name1, 'Radiohead', **{name2: 3})
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname, encoding='latin-1')
        H = nx.read_edgelist(fname, encoding='latin-1')
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_graph(self):
        G = self.G
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname)
        H2 = nx.read_edgelist(fname)
        assert H is not H2
        G.remove_node('g')
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_digraph(self):
        G = self.DG
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, create_using=nx.DiGraph())
        H2 = nx.read_edgelist(fname, create_using=nx.DiGraph())
        assert H is not H2
        G.remove_node('g')
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_integers(self):
        G = nx.convert_node_labels_to_integers(self.G)
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int)
        G.remove_nodes_from(list(nx.isolates(G)))
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_multigraph(self):
        G = self.XG
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        H2 = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_multidigraph(self):
        G = self.XDG
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        H2 = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)