import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_pandas_adjacency_with_nodelist():
    G = nx.complete_graph(5)
    nodelist = [1, 4]
    expected = pd.DataFrame([[0, 1], [1, 0]], dtype=int, index=nodelist, columns=nodelist)
    pd.testing.assert_frame_equal(expected, nx.to_pandas_adjacency(G, nodelist, dtype=int))