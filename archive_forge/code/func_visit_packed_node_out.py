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
def visit_packed_node_out(self, node):
    graph_node_id = str(id(node))
    graph_node = self.graph.get_node(graph_node_id)[0]
    for child in [node.left, node.right]:
        if child is not None:
            child_graph_node_id = str(id(child.token if isinstance(child, TokenNode) else child))
            child_graph_node = self.graph.get_node(child_graph_node_id)[0]
            self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node))
        else:
            child_graph_node_id = str(randint(100000000000000000000000000000, 123456789012345678901234567890))
            child_graph_node_style = 'invis'
            child_graph_node = self.pydot.Node(child_graph_node_id, style=child_graph_node_style, label='None')
            child_edge_style = 'invis'
            self.graph.add_node(child_graph_node)
            self.graph.add_edge(self.pydot.Edge(graph_node, child_graph_node, style=child_edge_style))