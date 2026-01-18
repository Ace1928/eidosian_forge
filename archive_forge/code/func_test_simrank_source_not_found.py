import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_simrank_source_not_found(self):
    G = nx.cycle_graph(5)
    with pytest.raises(nx.NodeNotFound, match='Source node 10 not in G'):
        nx.simrank_similarity(G, source=10)