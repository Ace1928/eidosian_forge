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
class ForestSumVisitor(ForestVisitor):
    """
    A visitor for prioritizing ambiguous parts of the Forest.

    This visitor is used when support for explicit priorities on
    rules is requested (whether normal, or invert). It walks the
    forest (or subsets thereof) and cascades properties upwards
    from the leaves.

    It would be ideal to do this during parsing, however this would
    require processing each Earley item multiple times. That's
    a big performance drawback; so running a forest walk is the
    lesser of two evils: there can be significantly more Earley
    items created during parsing than there are SPPF nodes in the
    final tree.
    """

    def __init__(self):
        super(ForestSumVisitor, self).__init__(single_visit=True)

    def visit_packed_node_in(self, node):
        yield node.left
        yield node.right

    def visit_symbol_node_in(self, node):
        return iter(node.children)

    def visit_packed_node_out(self, node):
        priority = node.rule.options.priority if not node.parent.is_intermediate and node.rule.options.priority else 0
        priority += getattr(node.right, 'priority', 0)
        priority += getattr(node.left, 'priority', 0)
        node.priority = priority

    def visit_symbol_node_out(self, node):
        node.priority = max((child.priority for child in node.children))