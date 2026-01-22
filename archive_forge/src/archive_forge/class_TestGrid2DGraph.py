from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestGrid2DGraph:
    """Unit tests for :func:`networkx.generators.lattice.grid_2d_graph`"""

    def test_number_of_vertices(self):
        m, n = (5, 6)
        G = nx.grid_2d_graph(m, n)
        assert len(G) == m * n

    def test_degree_distribution(self):
        m, n = (5, 6)
        G = nx.grid_2d_graph(m, n)
        expected_histogram = [0, 0, 4, 2 * (m + n) - 8, (m - 2) * (n - 2)]
        assert nx.degree_histogram(G) == expected_histogram

    def test_directed(self):
        m, n = (5, 6)
        G = nx.grid_2d_graph(m, n)
        H = nx.grid_2d_graph(m, n, create_using=nx.DiGraph())
        assert H.succ == G.adj
        assert H.pred == G.adj

    def test_multigraph(self):
        m, n = (5, 6)
        G = nx.grid_2d_graph(m, n)
        H = nx.grid_2d_graph(m, n, create_using=nx.MultiGraph())
        assert list(H.edges()) == list(G.edges())

    def test_periodic(self):
        G = nx.grid_2d_graph(0, 0, periodic=True)
        assert dict(G.degree()) == {}
        for m, n, H in [(2, 2, nx.cycle_graph(4)), (1, 7, nx.cycle_graph(7)), (7, 1, nx.cycle_graph(7)), (2, 5, nx.circular_ladder_graph(5)), (5, 2, nx.circular_ladder_graph(5)), (2, 4, nx.cubical_graph()), (4, 2, nx.cubical_graph())]:
            G = nx.grid_2d_graph(m, n, periodic=True)
            assert nx.could_be_isomorphic(G, H)

    def test_periodic_iterable(self):
        m, n = (3, 7)
        for a, b in product([0, 1], [0, 1]):
            G = nx.grid_2d_graph(m, n, periodic=(a, b))
            assert G.number_of_nodes() == m * n
            assert G.number_of_edges() == (m + a - 1) * n + (n + b - 1) * m

    def test_periodic_directed(self):
        G = nx.grid_2d_graph(4, 2, periodic=True)
        H = nx.grid_2d_graph(4, 2, periodic=True, create_using=nx.DiGraph())
        assert H.succ == G.adj
        assert H.pred == G.adj

    def test_periodic_multigraph(self):
        G = nx.grid_2d_graph(4, 2, periodic=True)
        H = nx.grid_2d_graph(4, 2, periodic=True, create_using=nx.MultiGraph())
        assert list(G.edges()) == list(H.edges())

    def test_exceptions(self):
        pytest.raises(nx.NetworkXError, nx.grid_2d_graph, -3, 2)
        pytest.raises(nx.NetworkXError, nx.grid_2d_graph, 3, -2)
        pytest.raises(TypeError, nx.grid_2d_graph, 3.3, 2)
        pytest.raises(TypeError, nx.grid_2d_graph, 3, 2.2)

    def test_node_input(self):
        G = nx.grid_2d_graph(4, 2, periodic=True)
        H = nx.grid_2d_graph(range(4), range(2), periodic=True)
        assert nx.is_isomorphic(H, G)
        H = nx.grid_2d_graph('abcd', 'ef', periodic=True)
        assert nx.is_isomorphic(H, G)
        G = nx.grid_2d_graph(5, 6)
        H = nx.grid_2d_graph(range(5), range(6))
        assert edges_equal(H, G)