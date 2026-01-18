from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def whitehead_move(self, a, cut_set):
    """
        Return a presentation obtained by performing a Whitehead move
        (T-transformation) on this presentation.
        """
    new_relators = []
    for relator in self.relators:
        new_relator = []
        for x in relator:
            if x == a or x == -a:
                new_relator.append(x)
            else:
                if -x not in cut_set:
                    new_relator.append(a)
                new_relator.append(x)
                if x not in cut_set:
                    new_relator.append(-a)
        W = CyclicWord(new_relator, self.alphabet)
        new_relators.append(W)
    return Presentation(new_relators, self.generators)