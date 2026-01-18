import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def topological_numbering(G):
    """
    Finds an optimal weighted topological numbering a directed acyclic graph
    which doesn't have any local moves which decrease the lengths of the
    (non-dummy) edges.
    """
    n = basic_topological_numbering(G)
    success = True
    while success:
        success = False
        for v in G.vertices:
            below = len([e for e in G.incoming(v) if e.dummy is False])
            above = len([e for e in G.outgoing(v) if e.dummy is False])
            if above != below:
                if above > below:
                    new_pos = min((n[e.head] for e in G.outgoing(v))) - 1
                else:
                    new_pos = max((n[e.tail] for e in G.incoming(v))) + 1
                if new_pos != n[v]:
                    n[v] = new_pos
                    success = True
    return n