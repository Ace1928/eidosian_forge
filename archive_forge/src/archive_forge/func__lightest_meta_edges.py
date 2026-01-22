import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _lightest_meta_edges(mapping, avail_uv, avail_w):
    """Maps available edges in the original graph to edges in the metagraph.

    Parameters
    ----------
    mapping : dict
        mapping produced by :func:`collapse`, that maps each node in the
        original graph to a node in the meta graph

    avail_uv : list
        list of edges

    avail_w : list
        list of edge weights

    Notes
    -----
    Each node in the metagraph is a k-edge-connected component in the original
    graph.  We don't care about any edge within the same k-edge-connected
    component, so we ignore self edges.  We also are only interested in the
    minimum weight edge bridging each k-edge-connected component so, we group
    the edges by meta-edge and take the lightest in each group.

    Examples
    --------
    >>> # Each group represents a meta-node
    >>> groups = ([1, 2, 3], [4, 5], [6])
    >>> mapping = {n: meta_n for meta_n, ns in enumerate(groups) for n in ns}
    >>> avail_uv = [(1, 2), (3, 6), (1, 4), (5, 2), (6, 1), (2, 6), (3, 1)]
    >>> avail_w = [20, 99, 20, 15, 50, 99, 20]
    >>> sorted(_lightest_meta_edges(mapping, avail_uv, avail_w))
    [MetaEdge(meta_uv=(0, 1), uv=(5, 2), w=15), MetaEdge(meta_uv=(0, 2), uv=(6, 1), w=50)]
    """
    grouped_wuv = defaultdict(list)
    for w, (u, v) in zip(avail_w, avail_uv):
        meta_uv = _ordered(mapping[u], mapping[v])
        grouped_wuv[meta_uv].append((w, u, v))
    for (mu, mv), choices_wuv in grouped_wuv.items():
        if mu != mv:
            w, u, v = min(choices_wuv)
            yield MetaEdge((mu, mv), (u, v), w)