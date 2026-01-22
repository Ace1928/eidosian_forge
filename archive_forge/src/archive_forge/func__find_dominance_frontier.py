import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_dominance_frontier(self):
    idom = self._idom
    preds_table = self._preds
    df = {u: set() for u in idom}
    for u in idom:
        if len(preds_table[u]) < 2:
            continue
        for v in preds_table[u]:
            while v != idom[u]:
                df[v].add(u)
                v = idom[v]
    return df