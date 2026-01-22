from __future__ import annotations
from collections import deque
from dask.core import istask, subs
def _top_level(net, term):
    return net._rewrite(term)