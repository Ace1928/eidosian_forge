from itertools import product
import networkx as nx
from networkx.utils import not_implemented_for
def _edges_cross_nodes(G, H):
    if G.is_multigraph():
        for u, v, k, d in G.edges(data=True, keys=True):
            for x in H:
                yield ((u, x), (v, x), k, d)
    else:
        for u, v, d in G.edges(data=True):
            for x in H:
                if H.is_multigraph():
                    yield ((u, x), (v, x), None, d)
                else:
                    yield ((u, x), (v, x), d)