import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class LongestChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries longer edges before
    shorter ones.  This sorting order results in a type of best-first
    search strategy.
    """

    def sort_queue(self, queue, chart):
        queue.sort(key=lambda edge: edge.length())