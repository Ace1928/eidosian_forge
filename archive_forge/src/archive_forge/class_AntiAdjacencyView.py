import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
class AntiAdjacencyView(AntiAtlasView):
    """An adjacency outer dict for AntiGraph"""

    def __init__(self, graph):
        self._graph = graph
        self._atlas = graph._adj

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._graph)

    def __getitem__(self, node):
        if node not in self._graph:
            raise KeyError(node)
        return self._graph.AntiAtlasView(self._graph, node)