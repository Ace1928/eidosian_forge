import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def update_remaining(self, u, v, aux_graph=None):
    if aux_graph is not None:
        aux_graph.remove_edge(u, v)
    if self.remaining_degree[u] == 1:
        del self.remaining_degree[u]
        if aux_graph is not None:
            aux_graph.remove_node(u)
    else:
        self.remaining_degree[u] -= 1
    if self.remaining_degree[v] == 1:
        del self.remaining_degree[v]
        if aux_graph is not None:
            aux_graph.remove_node(v)
    else:
        self.remaining_degree[v] -= 1