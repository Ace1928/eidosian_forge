import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_triangles():
    paths = [(11, 12, 13, 11), (21, 22, 23, 21), (11, 21)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    assert fset(nx.k_edge_components(G, k=1)) == fset(nx.k_edge_subgraphs(G, k=1))
    assert fset(nx.k_edge_components(G, k=2)) == fset(nx.k_edge_subgraphs(G, k=2))
    assert fset(nx.k_edge_components(G, k=3)) == fset(nx.k_edge_subgraphs(G, k=3))
    _check_edge_connectivity(G)