from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class InstantiateVarsChart(FeatureChart):
    """
    A specialized chart that 'instantiates' variables whose names
    start with '@', by replacing them with unique new variables.
    In particular, whenever a complete edge is added to the chart, any
    variables in the edge's ``lhs`` whose names start with '@' will be
    replaced by unique new ``Variable``.
    """

    def __init__(self, tokens):
        FeatureChart.__init__(self, tokens)

    def initialize(self):
        self._instantiated = set()
        FeatureChart.initialize(self)

    def insert(self, edge, child_pointer_list):
        if edge in self._instantiated:
            return False
        self.instantiate_edge(edge)
        return FeatureChart.insert(self, edge, child_pointer_list)

    def instantiate_edge(self, edge):
        """
        If the edge is a ``FeatureTreeEdge``, and it is complete,
        then instantiate all variables whose names start with '@',
        by replacing them with unique new variables.

        Note that instantiation is done in-place, since the
        parsing algorithms might already hold a reference to
        the edge for future use.
        """
        if not isinstance(edge, FeatureTreeEdge):
            return
        if not edge.is_complete():
            return
        if edge in self._edge_to_cpls:
            return
        inst_vars = self.inst_vars(edge)
        if not inst_vars:
            return
        self._instantiated.add(edge)
        edge._lhs = edge.lhs().substitute_bindings(inst_vars)

    def inst_vars(self, edge):
        return {var: logic.unique_variable() for var in edge.lhs().variables() if var.name.startswith('@')}