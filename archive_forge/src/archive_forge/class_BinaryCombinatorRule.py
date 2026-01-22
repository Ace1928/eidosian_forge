import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
class BinaryCombinatorRule(AbstractChartRule):
    """
    Class implementing application of a binary combinator to a chart.
    Takes the directed combinator to apply.
    """
    NUMEDGES = 2

    def __init__(self, combinator):
        self._combinator = combinator

    def apply(self, chart, grammar, left_edge, right_edge):
        if not left_edge.end() == right_edge.start():
            return
        if self._combinator.can_combine(left_edge.categ(), right_edge.categ()):
            for res in self._combinator.combine(left_edge.categ(), right_edge.categ()):
                new_edge = CCGEdge(span=(left_edge.start(), right_edge.end()), categ=res, rule=self._combinator)
                if chart.insert(new_edge, (left_edge, right_edge)):
                    yield new_edge

    def __str__(self):
        return '%s' % self._combinator