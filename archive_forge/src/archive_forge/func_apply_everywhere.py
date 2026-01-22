import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def apply_everywhere(self, chart, grammar):
    if self.NUM_EDGES == 0:
        yield from self.apply(chart, grammar)
    elif self.NUM_EDGES == 1:
        for e1 in chart:
            yield from self.apply(chart, grammar, e1)
    elif self.NUM_EDGES == 2:
        for e1 in chart:
            for e2 in chart:
                yield from self.apply(chart, grammar, e1, e2)
    elif self.NUM_EDGES == 3:
        for e1 in chart:
            for e2 in chart:
                for e3 in chart:
                    yield from self.apply(chart, grammar, e1, e2, e3)
    else:
        raise AssertionError('NUM_EDGES>3 is not currently supported')