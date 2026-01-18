from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph

        Computes the probability of a dependency graph based
        on the parser's probability model (defined by the parser's
        statistical dependency grammar).

        :param dg: A dependency graph to score.
        :type dg: DependencyGraph
        :return: The probability of the dependency graph.
        :rtype: int
        