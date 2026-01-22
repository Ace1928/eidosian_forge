import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@not_implemented_for('undirected')
@nx._dispatch
def antichains(G, topo_order=None):
    """Generates antichains from a directed acyclic graph (DAG).

    An antichain is a subset of a partially ordered set such that any
    two elements in the subset are incomparable.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    topo_order: list or tuple, optional
        A topological order for G (if None, the function will compute one)

    Yields
    ------
    antichain : list
        a list of nodes in `G` representing an antichain

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed

    NetworkXUnfeasible
        If `G` contains a cycle

    Examples
    --------
    >>> DG = nx.DiGraph([(1, 2), (1, 3)])
    >>> list(nx.antichains(DG))
    [[], [3], [2], [2, 3], [1]]

    Notes
    -----
    This function was originally developed by Peter Jipsen and Franco Saliola
    for the SAGE project. It's included in NetworkX with permission from the
    authors. Original SAGE code at:

    https://github.com/sagemath/sage/blob/master/src/sage/combinat/posets/hasse_diagram.py

    References
    ----------
    .. [1] Free Lattices, by R. Freese, J. Jezek and J. B. Nation,
       AMS, Vol 42, 1995, p. 226.
    """
    if topo_order is None:
        topo_order = list(nx.topological_sort(G))
    TC = nx.transitive_closure_dag(G, topo_order)
    antichains_stacks = [([], list(reversed(topo_order)))]
    while antichains_stacks:
        antichain, stack = antichains_stacks.pop()
        yield antichain
        while stack:
            x = stack.pop()
            new_antichain = antichain + [x]
            new_stack = [t for t in stack if not (t in TC[x] or x in TC[t])]
            antichains_stacks.append((new_antichain, new_stack))