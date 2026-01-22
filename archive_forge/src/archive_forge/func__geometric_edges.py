import math
from bisect import bisect_left
from itertools import accumulate, combinations, product
import networkx as nx
from networkx.utils import py_random_state
def _geometric_edges(G, radius, p, pos_name):
    """
    Implements `geometric_edges` without input validation. See `geometric_edges`
    for complete docstring.
    """
    nodes_pos = G.nodes(data=pos_name)
    try:
        import scipy as sp
    except ImportError:
        radius_p = radius ** p
        edges = [(u, v) for (u, pu), (v, pv) in combinations(nodes_pos, 2) if sum((abs(a - b) ** p for a, b in zip(pu, pv))) <= radius_p]
        return edges
    nodes, coords = list(zip(*nodes_pos))
    kdtree = sp.spatial.cKDTree(coords)
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = [(nodes[u], nodes[v]) for u, v in sorted(edge_indexes)]
    return edges