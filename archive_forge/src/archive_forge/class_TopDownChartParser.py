import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class TopDownChartParser(ChartParser):
    """
    A ``ChartParser`` using a top-down parsing strategy.
    See ``ChartParser`` for more information.
    """

    def __init__(self, grammar, **parser_args):
        ChartParser.__init__(self, grammar, TD_STRATEGY, **parser_args)