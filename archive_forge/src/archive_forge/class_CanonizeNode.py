from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
class CanonizeNode:

    def __init__(self, presentation, remaining, ordering=[]):
        self.presentation = presentation
        self.generators = presentation.generators
        self.relators = presentation.relators
        self.remaining = remaining
        self.ordering = ordering

    def __repr__(self):
        return '%s\n%s' % (self.presentation, self.ordering)

    def children(self):
        childlist = []
        least = Complexity()
        length = len(self.generators)
        for relator in self.remaining:
            complexity, minima = relator.minima(length, self.ordering)
            if complexity > least:
                continue
            if complexity < least:
                least = complexity
                childlist = []
            for minimum in minima:
                word, ordering = minimum
                relators = self.relators + [word]
                remaining = list(self.remaining)
                remaining.remove(relator)
                P = Presentation(relators, generators=self.generators)
                childlist.append(CanonizeNode(P, remaining, ordering))
        return childlist