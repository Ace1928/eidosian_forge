import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_zero_k_exception():
    G = nx.Graph()
    pytest.raises(ValueError, nx.k_edge_components, G, k=0)
    pytest.raises(ValueError, nx.k_edge_subgraphs, G, k=0)
    aux_graph = EdgeComponentAuxGraph.construct(G)
    pytest.raises(ValueError, list, aux_graph.k_edge_components(k=0))
    pytest.raises(ValueError, list, aux_graph.k_edge_subgraphs(k=0))
    pytest.raises(ValueError, list, general_k_edge_subgraphs(G, k=0))