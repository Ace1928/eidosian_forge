import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class RandomChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in random order.
    This sorting order results in a random search strategy.
    """

    def sort_queue(self, queue, chart):
        i = random.randint(0, len(queue) - 1)
        queue[-1], queue[i] = (queue[i], queue[-1])