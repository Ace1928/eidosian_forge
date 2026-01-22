import collections
from itertools import repeat
from typing import List, Dict, Any
def _ntuple(n, name='parse'):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    parse.__name__ = name
    return parse