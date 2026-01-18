import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def treewidth_min_degree(G):
    """Returns a treewidth decomposition using the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree, i.e., first
    the node with the lowest degree is chosen, then the graph is updated
    and the corresponding node is removed. Next, a new node with the lowest
    degree is chosen, and so on.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
          2-tuple with treewidth and the corresponding decomposed tree.
    """
    deg_heuristic = MinDegreeHeuristic(G)
    return treewidth_decomp(G, lambda graph: deg_heuristic.best_node(graph))