import networkx as nx
from networkx import is_strongly_regular
def test_not_connected(self):
    G = nx.cycle_graph(4)
    nx.add_cycle(G, [5, 6, 7])
    assert not nx.is_distance_regular(G)