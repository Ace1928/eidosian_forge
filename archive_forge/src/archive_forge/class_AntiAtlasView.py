import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
class AntiAtlasView(Mapping):
    """An adjacency inner dict for AntiGraph"""

    def __init__(self, graph, node):
        self._graph = graph
        self._atlas = graph._adj[node]
        self._node = node

    def __len__(self):
        return len(self._graph) - len(self._atlas) - 1

    def __iter__(self):
        return (n for n in self._graph if n not in self._atlas and n != self._node)

    def __getitem__(self, nbr):
        nbrs = set(self._graph._adj) - set(self._atlas) - {self._node}
        if nbr in nbrs:
            return self._graph.all_edge_dict
        raise KeyError(nbr)