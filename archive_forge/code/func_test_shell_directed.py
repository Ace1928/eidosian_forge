import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def test_shell_directed():
    seeds = [31]
    for seed in seeds:
        constructor = [(12, 70, 0.8), (15, 40, 0.6)]
        G = nx.random_shell_graph(constructor, seed=seed).to_directed()
        _check_edge_connectivity(G)