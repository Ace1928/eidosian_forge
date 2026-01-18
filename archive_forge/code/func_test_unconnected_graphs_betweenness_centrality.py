import pytest
import networkx as nx
from networkx import approximate_current_flow_betweenness_centrality as approximate_cfbc
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
@pytest.mark.parametrize('centrality_func', (nx.current_flow_betweenness_centrality, nx.edge_current_flow_betweenness_centrality, nx.approximate_current_flow_betweenness_centrality))
def test_unconnected_graphs_betweenness_centrality(centrality_func):
    G = nx.Graph([(1, 2), (3, 4)])
    G.add_node(5)
    with pytest.raises(nx.NetworkXError, match='Graph not connected'):
        centrality_func(G)