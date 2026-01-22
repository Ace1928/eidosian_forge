import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def _assert_subgraph_edge_connectivity(G, ccs_subgraph, k):
    """
    tests properties of k-edge-connected subgraphs

    the actual edge connectivity should be no less than k unless the cc is a
    single node.
    """
    for cc in ccs_subgraph:
        C = G.subgraph(cc)
        if len(cc) > 1:
            connectivity = nx.edge_connectivity(C)
            assert connectivity >= k