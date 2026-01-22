import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def _apply_complete(self, chart, grammar, right_edge):
    end = right_edge.end()
    nexttoken = end < chart.num_leaves() and chart.leaf(end)
    for left_edge in chart.select(end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()):
        if _bottomup_filter(grammar, nexttoken, left_edge.rhs(), left_edge.dot()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge