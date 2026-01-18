from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
def update_incumbent_if_improved(self, C, C_weight):
    """Update the incumbent if the node set C has greater weight.

        C is assumed to be a clique.
        """
    if C_weight > self.incumbent_weight:
        self.incumbent_nodes = C[:]
        self.incumbent_weight = C_weight