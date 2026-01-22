import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def _trace_new_edges(self, chart, rule, new_edges, trace, edge_width):
    if not trace:
        return
    print_rule_header = trace > 1
    for edge in new_edges:
        if print_rule_header:
            print('%s:' % rule)
            print_rule_header = False
        print(chart.pretty_format_edge(edge, edge_width))