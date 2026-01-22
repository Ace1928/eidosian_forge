from collections import Counter, defaultdict
from hashlib import blake2b
import networkx as nx
def _init_node_labels(G, edge_attr, node_attr):
    if node_attr:
        return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
    elif edge_attr:
        return {u: '' for u in G}
    else:
        return {u: str(deg) for u, deg in G.degree()}