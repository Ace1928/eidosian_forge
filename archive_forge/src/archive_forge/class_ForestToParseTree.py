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
class ForestToParseTree(ForestTransformer):
    """Used by the earley parser when ambiguity equals 'resolve' or
    'explicit'. Transforms an SPPF into an (ambiguous) parse tree.

    Parameters:
        tree_class: The tree class to use for construction
        callbacks: A dictionary of rules to functions that output a tree
        prioritizer: A ``ForestVisitor`` that manipulates the priorities of ForestNodes
        resolve_ambiguity: If True, ambiguities will be resolved based on
                        priorities. Otherwise, `_ambig` nodes will be in the resulting tree.
        use_cache: If True, the results of packed node transformations will be cached.
    """

    def __init__(self, tree_class=Tree, callbacks=dict(), prioritizer=ForestSumVisitor(), resolve_ambiguity=True, use_cache=True):
        super(ForestToParseTree, self).__init__()
        self.tree_class = tree_class
        self.callbacks = callbacks
        self.prioritizer = prioritizer
        self.resolve_ambiguity = resolve_ambiguity
        self._use_cache = use_cache
        self._cache = {}
        self._on_cycle_retreat = False
        self._cycle_node = None
        self._successful_visits = set()

    def visit(self, root):
        if self.prioritizer:
            self.prioritizer.visit(root)
        super(ForestToParseTree, self).visit(root)
        self._cache = {}

    def on_cycle(self, node, path):
        logger.debug('Cycle encountered in the SPPF at node: %s. As infinite ambiguities cannot be represented in a tree, this family of derivations will be discarded.', node)
        self._cycle_node = node
        self._on_cycle_retreat = True

    def _check_cycle(self, node):
        if self._on_cycle_retreat:
            if id(node) == id(self._cycle_node) or id(node) in self._successful_visits:
                self._cycle_node = None
                self._on_cycle_retreat = False
            else:
                return Discard

    def _collapse_ambig(self, children):
        new_children = []
        for child in children:
            if hasattr(child, 'data') and child.data == '_ambig':
                new_children += child.children
            else:
                new_children.append(child)
        return new_children

    def _call_rule_func(self, node, data):
        return self.callbacks[node.rule](data)

    def _call_ambig_func(self, node, data):
        if len(data) > 1:
            return self.tree_class('_ambig', data)
        elif data:
            return data[0]
        return Discard

    def transform_symbol_node(self, node, data):
        if id(node) not in self._successful_visits:
            return Discard
        r = self._check_cycle(node)
        if r is Discard:
            return r
        self._successful_visits.remove(id(node))
        data = self._collapse_ambig(data)
        return self._call_ambig_func(node, data)

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

    def visit_symbol_node_in(self, node):
        super(ForestToParseTree, self).visit_symbol_node_in(node)
        if self._on_cycle_retreat:
            return
        return node.children

    def visit_packed_node_in(self, node):
        self._on_cycle_retreat = False
        to_visit = super(ForestToParseTree, self).visit_packed_node_in(node)
        if not self.resolve_ambiguity or id(node.parent) not in self._successful_visits:
            if not self._use_cache or id(node) not in self._cache:
                return to_visit

    def visit_packed_node_out(self, node):
        super(ForestToParseTree, self).visit_packed_node_out(node)
        if not self._on_cycle_retreat:
            self._successful_visits.add(id(node.parent))