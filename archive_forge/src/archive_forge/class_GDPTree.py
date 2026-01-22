from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
class GDPTree:
    """
    Stores a forest representing the hierarchy between GDP components on a
    model: for single-level GDPs, each tree is rooted at a Disjunction and
    each of the Disjuncts in the Disjunction is a leaf. For nested GDPs, the
    Disjuncts may not be leaves, and could have child Disjunctions of their
    own.
    """

    def __init__(self):
        self._children = {}
        self._in_degrees = {}
        self._parent = {}
        self._root_disjunct = {}
        self._vertices = OrderedSet()

    @property
    def vertices(self):
        return self._vertices

    def add_node(self, u):
        self._vertices.add(u)

    def parent(self, u):
        """Returns the parent node of u, or None if u is a root.

        Arg:
            u : A node in the tree
        """
        if u not in self._vertices:
            raise ValueError("'%s' is not a vertex in the GDP tree. Cannot retrieve its parent." % u)
        if u in self._parent:
            return self._parent[u]
        else:
            return None

    def children(self, u):
        """Returns the direct descendents of node u.

        Arg:
            u : A node in the tree
        """
        return self._children[u]

    def parent_disjunct(self, u):
        """Returns the parent Disjunct of u, or None if u is the
        closest-to-root Disjunct in the forest.

        Arg:
            u : A node in the forest
        """
        if u.ctype is Disjunct:
            return self.parent(self.parent(u))
        else:
            return self.parent(u)

    def root_disjunct(self, u):
        """Returns the highest parent Disjunct in the hierarchy, or None if
        the component is not nested.

        Arg:
            u : A node in the tree
        """
        rootmost_disjunct = None
        parent = self.parent(u)
        while True:
            if parent is None:
                return rootmost_disjunct
            if parent.ctype is Disjunct:
                rootmost_disjunct = parent
            parent = self.parent(parent)

    def add_node(self, u):
        if u not in self._children:
            self._children[u] = OrderedSet()
        self._vertices.add(u)

    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self._children[u].add(v)
        if v in self._parent and self._parent[v] is not u:
            _raise_disjunct_in_multiple_disjunctions_error(v, u)
        self._parent[v] = u

    def _visit_vertex(self, u, leaf_to_root):
        if u in self._children:
            for v in self._children[u]:
                if v not in leaf_to_root:
                    self._visit_vertex(v, leaf_to_root)
        leaf_to_root.add(u)

    def _reverse_topological_iterator(self):
        leaf_to_root = OrderedSet()
        for u in self.vertices:
            if u not in leaf_to_root:
                self._visit_vertex(u, leaf_to_root)
        return leaf_to_root

    def topological_sort(self):
        return list(reversed(self._reverse_topological_iterator()))

    def reverse_topological_sort(self):
        return self._reverse_topological_iterator()

    def in_degree(self, u):
        if u not in self._parent:
            return 0
        return 1

    def is_leaf(self, u):
        if len(self._children[u]) == 0:
            return True
        return False

    @property
    def leaves(self):
        for u, children in self._children.items():
            if len(children) == 0:
                yield u

    @property
    def disjunct_nodes(self):
        for v in self._vertices:
            if v.ctype is Disjunct:
                yield v