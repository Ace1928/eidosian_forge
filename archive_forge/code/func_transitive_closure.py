import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch(preserve_all_attrs=True)
def transitive_closure(G, reflexive=False):
    """Returns transitive closure of a graph

    The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
    for all v, w in V there is an edge (v, w) in E+ if and only if there
    is a path from v to w in G.

    Handling of paths from v to v has some flexibility within this definition.
    A reflexive transitive closure creates a self-loop for the path
    from v to v of length 0. The usual transitive closure creates a
    self-loop only if a cycle exists (a path from v to v with length > 0).
    We also allow an option for no self-loops.

    Parameters
    ----------
    G : NetworkX Graph
        A directed/undirected graph/multigraph.
    reflexive : Bool or None, optional (default: False)
        Determines when cycles create self-loops in the Transitive Closure.
        If True, trivial cycles (length 0) create self-loops. The result
        is a reflexive transitive closure of G.
        If False (the default) non-trivial cycles create self-loops.
        If None, self-loops are not created.

    Returns
    -------
    NetworkX graph
        The transitive closure of `G`

    Raises
    ------
    NetworkXError
        If `reflexive` not in `{None, True, False}`

    Examples
    --------
    The treatment of trivial (i.e. length 0) cycles is controlled by the
    `reflexive` parameter.

    Trivial (i.e. length 0) cycles do not create self-loops when
    ``reflexive=False`` (the default)::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> TC = nx.transitive_closure(DG, reflexive=False)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (2, 3)])

    However, nontrivial (i.e. length greater than 0) cycles create self-loops
    when ``reflexive=False`` (the default)::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        >>> TC = nx.transitive_closure(DG, reflexive=False)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (1, 1), (2, 3), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)])

    Trivial cycles (length 0) create self-loops when ``reflexive=True``::

        >>> DG = nx.DiGraph([(1, 2), (2, 3)])
        >>> TC = nx.transitive_closure(DG, reflexive=True)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 1), (1, 3), (2, 3), (2, 2), (3, 3)])

    And the third option is not to create self-loops at all when ``reflexive=None``::

        >>> DG = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        >>> TC = nx.transitive_closure(DG, reflexive=None)
        >>> TC.edges()
        OutEdgeView([(1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)])

    References
    ----------
    .. [1] https://www.ics.uci.edu/~eppstein/PADS/PartialOrder.py
    """
    TC = G.copy()
    if reflexive not in {None, True, False}:
        raise nx.NetworkXError('Incorrect value for the parameter `reflexive`')
    for v in G:
        if reflexive is None:
            TC.add_edges_from(((v, u) for u in nx.descendants(G, v) if u not in TC[v]))
        elif reflexive is True:
            TC.add_edges_from(((v, u) for u in nx.descendants(G, v) | {v} if u not in TC[v]))
        elif reflexive is False:
            TC.add_edges_from(((v, e[1]) for e in nx.edge_bfs(G, v) if e[1] not in TC[v]))
    return TC