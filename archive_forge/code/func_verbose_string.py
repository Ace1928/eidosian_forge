from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def verbose_string(self, separator=' * '):
    ans = []
    alpha = self.alphabet
    for g, e in self.syllables():
        part = alpha[g] if e == 1 else alpha[g] + '^' + repr(e)
        ans.append(part)
    return separator.join(ans)