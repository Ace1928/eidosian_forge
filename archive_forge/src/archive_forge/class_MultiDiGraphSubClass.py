from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
class MultiDiGraphSubClass(nx.MultiDiGraph):
    node_dict_factory = CustomDictClass
    node_attr_dict_factory = CustomDictClass
    adjlist_outer_dict_factory = CustomDictClass
    adjlist_inner_dict_factory = CustomDictClass
    edge_key_dict_factory = CustomDictClass
    edge_attr_dict_factory = CustomDictClass
    graph_attr_dict_factory = CustomDictClass