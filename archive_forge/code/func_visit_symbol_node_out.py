from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
def visit_symbol_node_out(self, node):
    graph_node_id = str(id(node))
    graph_node = self.graph.get_node(graph_node_id)[0]
    for child in node.children:
        child_graph_node_id = str(id(child))
        child_graph_node = self.graph.get_node(child_graph_node_id)[0]
        self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))