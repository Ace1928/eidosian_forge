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
def transform_intermediate_node(self, node, data):
    if id(node) not in self._successful_visits:
        return Discard
    r = self._check_cycle(node)
    if r is Discard:
        return r
    self._successful_visits.remove(id(node))
    if len(data) > 1:
        children = [self.tree_class('_inter', c) for c in data]
        return self.tree_class('_iambig', children)
    return data[0]