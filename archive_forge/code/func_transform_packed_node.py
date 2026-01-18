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
def transform_packed_node(self, node, data):
    r = self._check_cycle(node)
    if r is Discard:
        return r
    if self.resolve_ambiguity and id(node.parent) in self._successful_visits:
        return Discard
    if self._use_cache and id(node) in self._cache:
        return self._cache[id(node)]
    children = []
    assert len(data) <= 2
    data = PackedData(node, data)
    if data.left is not PackedData.NO_DATA:
        if node.left.is_intermediate and isinstance(data.left, list):
            children += data.left
        else:
            children.append(data.left)
    if data.right is not PackedData.NO_DATA:
        children.append(data.right)
    if node.parent.is_intermediate:
        return self._cache.setdefault(id(node), children)
    return self._cache.setdefault(id(node), self._call_rule_func(node, children))