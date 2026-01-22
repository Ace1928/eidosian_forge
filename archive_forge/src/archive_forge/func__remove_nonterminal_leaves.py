from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _remove_nonterminal_leaves(G, terminals):
    terminals_set = set(terminals)
    for n in list(G.nodes):
        if n not in terminals_set and G.degree(n) == 1:
            G.remove_node(n)