import networkx as nx
from networkx.algorithms.approximation import average_clustering
def test_tetrahedral():
    G = nx.tetrahedral_graph()
    assert average_clustering(G, trials=len(G) // 2) == nx.average_clustering(G)