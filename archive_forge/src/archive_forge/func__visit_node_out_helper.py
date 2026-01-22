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
def _visit_node_out_helper(self, node, method):
    self.node_stack.pop()
    transformed = method(node, self.data[id(node)])
    if transformed is not Discard:
        self.data[self.node_stack[-1]].append(transformed)
    del self.data[id(node)]