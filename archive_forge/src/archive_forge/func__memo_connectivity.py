import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def _memo_connectivity(G, u, v, memo):
    edge = (u, v)
    if edge in memo:
        return memo[edge]
    if not G.is_directed():
        redge = (v, u)
        if redge in memo:
            return memo[redge]
    memo[edge] = nx.edge_connectivity(G, *edge)
    return memo[edge]