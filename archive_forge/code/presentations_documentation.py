from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator

        Return the relators of a canonized presentation as a tuple
        of tuples.  The result is hashable, but can be used to
        generate a canonical presentation equivalent to this one.
        