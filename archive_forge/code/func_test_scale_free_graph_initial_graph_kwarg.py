import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
@pytest.mark.parametrize('ig', (nx.Graph(), nx.DiGraph([(0, 1)])))
def test_scale_free_graph_initial_graph_kwarg(ig):
    with pytest.raises(nx.NetworkXError):
        scale_free_graph(100, initial_graph=ig)