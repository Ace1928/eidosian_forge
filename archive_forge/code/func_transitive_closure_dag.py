import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@not_implemented_for('undirected')
@nx._dispatch(preserve_all_attrs=True)
def transitive_closure_dag(G, topo_order=None):
    """Returns the transitive closure of a directed acyclic graph.

    This function is faster than the function `transitive_closure`, but fails
    if the graph has a cycle.

    The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
    for all v, w in V there is an edge (v, w) in E+ if and only if there
    is a non-null path from v to w in G.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    topo_order: list or tuple, optional
        A topological order for G (if None, the function will compute one)

    Returns
    -------
    NetworkX DiGraph
        The transitive closure of `G`

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed
    NetworkXUnfeasible
        If `G` has a cycle

    Examples
    --------
    >>> DG = nx.DiGraph([(1, 2), (2, 3)])
    >>> TC = nx.transitive_closure_dag(DG)
    >>> TC.edges()
    OutEdgeView([(1, 2), (1, 3), (2, 3)])

    Notes
    -----
    This algorithm is probably simple enough to be well-known but I didn't find
    a mention in the literature.
    """
    if topo_order is None:
        topo_order = list(topological_sort(G))
    TC = G.copy()
    for v in reversed(topo_order):
        TC.add_edges_from(((v, u) for u in nx.descendants_at_distance(TC, v, 2)))
    return TC