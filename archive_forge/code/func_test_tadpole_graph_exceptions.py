import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_tadpole_graph_exceptions(self):
    pytest.raises(nx.NetworkXError, nx.tadpole_graph, -1, 3)
    pytest.raises(nx.NetworkXError, nx.tadpole_graph, 0, 3)
    pytest.raises(nx.NetworkXError, nx.tadpole_graph, 1, 3)
    pytest.raises(nx.NetworkXError, nx.tadpole_graph, 5, -2)
    with pytest.raises(nx.NetworkXError):
        nx.tadpole_graph(2, 20, create_using=nx.DiGraph)
    with pytest.raises(nx.NetworkXError):
        nx.tadpole_graph(2, 20, create_using=nx.MultiDiGraph)