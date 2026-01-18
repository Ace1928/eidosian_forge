import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree

        Sort the given queue of edges, in descending order of the
        inside probabilities of the edges' trees.

        :param queue: The queue of ``Edge`` objects to sort.  Each edge in
            this queue is an edge that could be added to the chart by
            the fundamental rule; but that has not yet been added.
        :type queue: list(Edge)
        :param chart: The chart being used to parse the text.  This
            chart can be used to provide extra information for sorting
            the queue.
        :type chart: Chart
        :rtype: None
        