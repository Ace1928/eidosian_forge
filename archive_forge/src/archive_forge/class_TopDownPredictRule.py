import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class TopDownPredictRule(AbstractChartRule):
    """
    A rule licensing edges corresponding to the grammar productions
    for the nonterminal following an incomplete edge's dot.  In
    particular, this rule specifies that
    ``[A -> alpha \\* B beta][i:j]`` licenses the edge
    ``[B -> \\* gamma][j:j]`` for each grammar production ``B -> gamma``.

    :note: This rule corresponds to the Predictor Rule in Earley parsing.
    """
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_complete():
            return
        for prod in grammar.productions(lhs=edge.nextsym()):
            new_edge = TreeEdge.from_production(prod, edge.end())
            if chart.insert(new_edge, ()):
                yield new_edge