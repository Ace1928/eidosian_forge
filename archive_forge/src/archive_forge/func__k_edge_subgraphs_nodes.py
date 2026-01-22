import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def _k_edge_subgraphs_nodes(G, k):
    """Helper to get the nodes from the subgraphs.

    This allows k_edge_subgraphs to return a generator.
    """
    for C in general_k_edge_subgraphs(G, k):
        yield set(C.nodes())