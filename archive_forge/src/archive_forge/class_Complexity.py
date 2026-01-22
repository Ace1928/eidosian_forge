from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
class Complexity(list):

    def __lt__(self, other):
        if len(self) == len(other):
            return list.__lt__(self, other)
        else:
            return len(self) > len(other)

    def __gt__(self, other):
        if len(self) == len(other):
            return list.__gt__(self, other)
        else:
            return len(other) > len(self)

    def __le__(self, other):
        return not self > other

    def __ge__(self, other):
        return not self < other