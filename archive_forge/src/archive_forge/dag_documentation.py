import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
Iterate through the graph to compute all v-structures.

    V-structures are triples in the directed graph where
    two parent nodes point to the same child and the two parent nodes
    are not adjacent.

    Parameters
    ----------
    G : graph
        A networkx DiGraph.

    Returns
    -------
    vstructs : iterator of tuples
        The v structures within the graph. Each v structure is a 3-tuple with the
        parent, collider, and other parent.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (0, 5), (3, 1), (2, 4), (3, 1), (4, 5), (1, 5)])
    >>> sorted(nx.compute_v_structures(G))
    [(0, 5, 1), (0, 5, 4), (1, 5, 4)]

    Notes
    -----
    https://en.wikipedia.org/wiki/Collider_(statistics)
    