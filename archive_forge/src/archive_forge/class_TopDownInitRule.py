import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class TopDownInitRule(AbstractChartRule):
    """
    A rule licensing edges corresponding to the grammar productions for
    the grammar's start symbol.  In particular, this rule specifies that
    ``[S -> \\* alpha][0:i]`` is licensed for each grammar production
    ``S -> alpha``, where ``S`` is the grammar's start symbol.
    """
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for prod in grammar.productions(lhs=grammar.start()):
            new_edge = TreeEdge.from_production(prod, 0)
            if chart.insert(new_edge, ()):
                yield new_edge