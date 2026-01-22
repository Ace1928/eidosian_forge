from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureFundamentalRule(FundamentalRule):
    """
    A specialized version of the fundamental rule that operates on
    nonterminals whose symbols are ``FeatStructNonterminal``s.  Rather
    than simply comparing the nonterminals for equality, they are
    unified.  Variable bindings from these unifications are collected
    and stored in the chart using a ``FeatureTreeEdge``.  When a
    complete edge is generated, these bindings are applied to all
    nonterminals in the edge.

    The fundamental rule states that:

    - ``[A -> alpha \\* B1 beta][i:j]``
    - ``[B2 -> gamma \\*][j:k]``

    licenses the edge:

    - ``[A -> alpha B3 \\* beta][i:j]``

    assuming that B1 and B2 can be unified to generate B3.
    """

    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start() and left_edge.is_incomplete() and right_edge.is_complete() and isinstance(left_edge, FeatureTreeEdge)):
            return
        found = right_edge.lhs()
        nextsym = left_edge.nextsym()
        if isinstance(right_edge, FeatureTreeEdge):
            if not is_nonterminal(nextsym):
                return
            if left_edge.nextsym()[TYPE] != right_edge.lhs()[TYPE]:
                return
            bindings = left_edge.bindings()
            found = found.rename_variables(used_vars=left_edge.variables())
            result = unify(nextsym, found, bindings, rename_vars=False)
            if result is None:
                return
        else:
            if nextsym != found:
                return
            bindings = left_edge.bindings()
        new_edge = left_edge.move_dot_forward(right_edge.end(), bindings)
        if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
            yield new_edge