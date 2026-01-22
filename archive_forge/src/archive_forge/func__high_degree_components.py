import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def _high_degree_components(G, k):
    """Helper for filtering components that can't be k-edge-connected.

    Removes and generates each node with degree less than k.  Then generates
    remaining components where all nodes have degree at least k.
    """
    H = G.copy()
    singletons = set(_low_degree_nodes(H, k))
    while singletons:
        nbunch = set(it.chain.from_iterable(map(H.neighbors, singletons)))
        nbunch.difference_update(singletons)
        H.remove_nodes_from(singletons)
        for node in singletons:
            yield {node}
        singletons = set(_low_degree_nodes(H, k, nbunch))
    if G.is_directed():
        yield from nx.strongly_connected_components(H)
    else:
        yield from nx.connected_components(H)