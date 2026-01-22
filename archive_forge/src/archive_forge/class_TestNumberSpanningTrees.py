import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestNumberSpanningTrees:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')
        sp = pytest.importorskip('scipy')

    def test_nst_disconnected(self):
        G = nx.empty_graph(2)
        assert np.isclose(nx.number_of_spanning_trees(G), 0)

    def test_nst_no_nodes(self):
        G = nx.Graph()
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.number_of_spanning_trees(G)

    def test_nst_weight(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G.add_edge(1, 3, weight=1)
        G.add_edge(2, 3, weight=2)
        assert np.isclose(nx.number_of_spanning_trees(G), 3)
        assert np.isclose(nx.number_of_spanning_trees(G, weight='weight'), 5)

    def test_nst_negative_weight(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G.add_edge(1, 3, weight=-1)
        G.add_edge(2, 3, weight=-2)
        assert np.isclose(nx.number_of_spanning_trees(G), 3)
        assert np.isclose(nx.number_of_spanning_trees(G, weight='weight'), -1)

    def test_nst_selfloop(self):
        G = nx.complete_graph(3)
        G.add_edge(1, 1)
        assert np.isclose(nx.number_of_spanning_trees(G), 3)

    def test_nst_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge(1, 2)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 3)
        assert np.isclose(nx.number_of_spanning_trees(G), 5)

    def test_nst_complete_graph(self):
        N = 5
        G = nx.complete_graph(N)
        assert np.isclose(nx.number_of_spanning_trees(G), N ** (N - 2))

    def test_nst_path_graph(self):
        G = nx.path_graph(5)
        assert np.isclose(nx.number_of_spanning_trees(G), 1)

    def test_nst_cycle_graph(self):
        G = nx.cycle_graph(5)
        assert np.isclose(nx.number_of_spanning_trees(G), 5)

    def test_nst_directed_noroot(self):
        G = nx.empty_graph(3, create_using=nx.MultiDiGraph)
        with pytest.raises(nx.NetworkXError):
            nx.number_of_spanning_trees(G)

    def test_nst_directed_root_not_exist(self):
        G = nx.empty_graph(3, create_using=nx.MultiDiGraph)
        with pytest.raises(nx.NetworkXError):
            nx.number_of_spanning_trees(G, root=42)

    def test_nst_directed_not_weak_connected(self):
        G = nx.DiGraph()
        G.add_edge(1, 2)
        G.add_edge(3, 4)
        assert np.isclose(nx.number_of_spanning_trees(G, root=1), 0)

    def test_nst_directed_cycle_graph(self):
        G = nx.DiGraph()
        G = nx.cycle_graph(7, G)
        assert np.isclose(nx.number_of_spanning_trees(G, root=0), 1)

    def test_nst_directed_complete_graph(self):
        G = nx.DiGraph()
        G = nx.complete_graph(7, G)
        assert np.isclose(nx.number_of_spanning_trees(G, root=0), 7 ** 5)

    def test_nst_directed_multi(self):
        G = nx.MultiDiGraph()
        G = nx.cycle_graph(3, G)
        G.add_edge(1, 2)
        assert np.isclose(nx.number_of_spanning_trees(G, root=0), 2)

    def test_nst_directed_selfloop(self):
        G = nx.MultiDiGraph()
        G = nx.cycle_graph(3, G)
        G.add_edge(1, 1)
        assert np.isclose(nx.number_of_spanning_trees(G, root=0), 1)

    def test_nst_directed_weak_connected(self):
        G = nx.MultiDiGraph()
        G = nx.cycle_graph(3, G)
        G.remove_edge(1, 2)
        assert np.isclose(nx.number_of_spanning_trees(G, root=0), 0)

    def test_nst_directed_weighted(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=1)
        G.add_edge(2, 3, weight=3)
        Nst = nx.number_of_spanning_trees(G, root=1, weight='weight')
        assert np.isclose(Nst, 8)
        Nst = nx.number_of_spanning_trees(G, root=2, weight='weight')
        assert np.isclose(Nst, 0)
        Nst = nx.number_of_spanning_trees(G, root=3, weight='weight')
        assert np.isclose(Nst, 0)