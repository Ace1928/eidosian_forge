import networkx as nx
from collections import deque
class MultiEdge(BaseEdge):
    """
    An undirected edge.  MultiEdges are equal if they have the
    same vertices.  The multiplicity is initialized to 1.
    """

    def __init__(self, x, y):
        self.multiplicity = 1

    def __repr__(self):
        return '%s --%d-- %s' % (self[0], self.multiplicity, self[1])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return set(self) == set(other)