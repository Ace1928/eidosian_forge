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
class SymbolNode(ForestNode):
    """
    A Symbol Node represents a symbol (or Intermediate LR0).

    Symbol nodes are keyed by the symbol (s). For intermediate nodes
    s will be an LR0, stored as a tuple of (rule, ptr). For completed symbol
    nodes, s will be a string representing the non-terminal origin (i.e.
    the left hand side of the rule).

    The children of a Symbol or Intermediate Node will always be Packed Nodes;
    with each Packed Node child representing a single derivation of a production.

    Hence a Symbol Node with a single child is unambiguous.

    Parameters:
        s: A Symbol, or a tuple of (rule, ptr) for an intermediate node.
        start: The index of the start of the substring matched by this symbol (inclusive).
        end: The index of the end of the substring matched by this symbol (exclusive).

    Properties:
        is_intermediate: True if this node is an intermediate node.
        priority: The priority of the node's symbol.
    """
    Set: Type[AbstractSet] = set
    __slots__ = ('s', 'start', 'end', '_children', 'paths', 'paths_loaded', 'priority', 'is_intermediate', '_hash')

    def __init__(self, s, start, end):
        self.s = s
        self.start = start
        self.end = end
        self._children = self.Set()
        self.paths = self.Set()
        self.paths_loaded = False
        self.priority = float('-inf')
        self.is_intermediate = isinstance(s, tuple)
        self._hash = hash((self.s, self.start, self.end))

    def add_family(self, lr0, rule, start, left, right):
        self._children.add(PackedNode(self, lr0, rule, start, left, right))

    def add_path(self, transitive, node):
        self.paths.add((transitive, node))

    def load_paths(self):
        for transitive, node in self.paths:
            if transitive.next_titem is not None:
                vn = type(self)(transitive.next_titem.s, transitive.next_titem.start, self.end)
                vn.add_path(transitive.next_titem, node)
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, vn)
            else:
                self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, node)
        self.paths_loaded = True

    @property
    def is_ambiguous(self):
        """Returns True if this node is ambiguous."""
        return len(self.children) > 1

    @property
    def children(self):
        """Returns a list of this node's children sorted from greatest to
        least priority."""
        if not self.paths_loaded:
            self.load_paths()
        return sorted(self._children, key=attrgetter('sort_key'))

    def __iter__(self):
        return iter(self._children)

    def __eq__(self, other):
        if not isinstance(other, SymbolNode):
            return False
        return self is other or (type(self.s) == type(other.s) and self.s == other.s and (self.start == other.start) and (self.end is other.end))

    def __hash__(self):
        return self._hash

    def __repr__(self):
        if self.is_intermediate:
            rule = self.s[0]
            ptr = self.s[1]
            before = (expansion.name for expansion in rule.expansion[:ptr])
            after = (expansion.name for expansion in rule.expansion[ptr:])
            symbol = '{} ::= {}* {}'.format(rule.origin.name, ' '.join(before), ' '.join(after))
        else:
            symbol = self.s.name
        return '({}, {}, {}, {})'.format(symbol, self.start, self.end, self.priority)