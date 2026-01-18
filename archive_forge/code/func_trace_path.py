from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def trace_path(self, p, w):
    """
        Returns the nodes and edges on the path from node p to its ancestor w.
        """
    Wn = [p]
    We = []
    while p != w:
        We.append(self.parent_edge[p])
        p = self.parent[p]
        Wn.append(p)
    return (Wn, We)