from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
def test_too_many_blocks():
    with pytest.raises(nx.NetworkXError):
        G = nx.barbell_graph(3, 0)
        partition = ({0, 1}, {2}, {3, 4, 5})
        kernighan_lin_bisection(G, partition)