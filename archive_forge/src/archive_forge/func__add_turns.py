import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def _add_turns(self):
    self.turns = turns = []
    for i, (e0, v0) in enumerate(self):
        e1, v1 = self[i + 1]
        if e0.kind == e1.kind:
            turns.append(0)
        else:
            t = (e0.tail == v0) ^ (e1.head == v0) ^ (e0.kind == 'horizontal')
            turns.append(-1 if t else 1)
    rotation = sum(turns)
    assert abs(rotation) == 4
    self.exterior = rotation == -4