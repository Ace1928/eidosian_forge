import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def treewidth_min_fill_in(G):
    """Returns a treewidth decomposition using the Minimum Fill-in heuristic.

    The heuristic chooses a node from the graph, where the number of edges
    added turning the neighbourhood of the chosen node into clique is as
    small as possible.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G, min_fill_in_heuristic)