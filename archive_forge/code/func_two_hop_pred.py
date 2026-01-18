import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def two_hop_pred(self, Gx, Gx_node, core_x, pred):
    """
        The predecessors of the ego node.
        """
    return all((self.one_hop(Gx, p, core_x, self.preds(Gx, core_x, p), self.succs(Gx, core_x, p, Gx_node)) for p in pred))