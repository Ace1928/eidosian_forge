import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
class newGraph(G.to_undirected_class()):

    def to_directed_class(self):
        return newDiGraph

    def to_undirected_class(self):
        return newGraph