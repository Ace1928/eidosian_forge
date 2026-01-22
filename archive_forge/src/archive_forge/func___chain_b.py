from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def __chain_b(self):
    b = self.b
    self.b2j = b2j = {}
    for i, elt in enumerate(b):
        indices = b2j.setdefault(elt, [])
        indices.append(i)
    self.bjunk = junk = set()
    isjunk = self.isjunk
    if isjunk:
        for elt in b2j.keys():
            if isjunk(elt):
                junk.add(elt)
        for elt in junk:
            del b2j[elt]
    self.bpopular = popular = set()
    n = len(b)
    if self.autojunk and n >= 200:
        ntest = n // 100 + 1
        for elt, idxs in b2j.items():
            if len(idxs) > ntest:
                popular.add(elt)
        for elt in popular:
            del b2j[elt]