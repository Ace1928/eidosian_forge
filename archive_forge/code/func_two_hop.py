import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def two_hop(self, Gx, core_x, Gx_node, neighbors):
    """
        Paths of length 2 from Gx_node should be time-respecting.
        """
    return all((self.one_hop(Gx, v, [n for n in Gx[v] if n in core_x] + [Gx_node]) for v in neighbors))