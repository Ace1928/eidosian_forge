import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
@pytest.mark.parametrize('method', (None, 'mehlhorn', 'kou'))
def test_steiner_tree_methods(method):
    G = nx.star_graph(4)
    expected = nx.Graph([(0, 1), (0, 3)])
    st = nx.approximation.steiner_tree(G, [1, 3], method=method)
    assert nx.utils.edges_equal(st.edges, expected.edges)