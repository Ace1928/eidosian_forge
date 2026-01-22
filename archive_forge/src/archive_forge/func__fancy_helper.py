from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _fancy_helper(self, a, alo, ahi, b, blo, bhi):
    g = []
    if alo < ahi:
        if blo < bhi:
            g = self._fancy_replace(a, alo, ahi, b, blo, bhi)
        else:
            g = self._dump('-', a, alo, ahi)
    elif blo < bhi:
        g = self._dump('+', b, blo, bhi)
    yield from g