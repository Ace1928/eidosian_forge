import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.cluster import cc_dot, cc_max, cc_min
def test_not_bipartite():
    with pytest.raises(nx.NetworkXError):
        bipartite.clustering(nx.complete_graph(4))