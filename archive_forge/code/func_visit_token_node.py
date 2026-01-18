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
def visit_token_node(self, node):
    graph_node_id = str(id(node))
    graph_node_label = '"{}"'.format(node.value.replace('"', '\\"'))
    graph_node_color = 8421504
    graph_node_style = '"filled,rounded"'
    graph_node_shape = 'diamond'
    graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
    self.graph.add_node(graph_node)