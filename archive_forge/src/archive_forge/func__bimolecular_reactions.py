import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def _bimolecular_reactions(self):
    A = [[None] * self.ns for _ in range(self.ns)]
    unconsidered_ri = set()
    for i, r in enumerate(self.rxns):
        if r.order() == 2:
            keys = [k for k, v in r.reac.items() if v != 0]
            if len(keys) == 1:
                ri = ci = self.as_substance_index(keys[0])
            elif len(keys) == 2:
                ri, ci = sorted(map(self.as_substance_index, keys))
            else:
                raise NotImplementedError('Need 1 or 2 keys')
            if A[ri][ci] is None:
                A[ri][ci] = list()
            A[ri][ci].append((i, r))
        else:
            unconsidered_ri.add(i)
    return (A, unconsidered_ri)