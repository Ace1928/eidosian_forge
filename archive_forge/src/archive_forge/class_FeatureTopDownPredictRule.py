from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureTopDownPredictRule(CachedTopDownPredictRule):
    """
    A specialized version of the (cached) top down predict rule that operates
    on nonterminals whose symbols are ``FeatStructNonterminal``.  Rather
    than simply comparing the nonterminals for equality, they are
    unified.

    The top down expand rule states that:

    - ``[A -> alpha \\* B1 beta][i:j]``

    licenses the edge:

    - ``[B2 -> \\* gamma][j:j]``

    for each grammar production ``B2 -> gamma``, assuming that B1
    and B2 can be unified.
    """

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            return
        nextsym, index = (edge.nextsym(), edge.end())
        if not is_nonterminal(nextsym):
            return
        nextsym_with_bindings = edge.next_with_bindings()
        done = self._done.get((nextsym_with_bindings, index), (None, None))
        if done[0] is chart and done[1] is grammar:
            return
        for prod in grammar.productions(lhs=nextsym):
            if prod.rhs():
                first = prod.rhs()[0]
                if is_terminal(first):
                    if index >= chart.num_leaves():
                        continue
                    if first != chart.leaf(index):
                        continue
            if unify(prod.lhs(), nextsym_with_bindings, rename_vars=True):
                new_edge = FeatureTreeEdge.from_production(prod, edge.end())
                if chart.insert(new_edge, ()):
                    yield new_edge
        self._done[nextsym_with_bindings, index] = (chart, grammar)