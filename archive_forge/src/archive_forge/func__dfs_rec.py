import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _dfs_rec(node):
    if node not in seen:
        seen.add(node)
        for dest in succs[node]:
            if (node, dest) not in back_edges:
                _dfs_rec(dest)
        post_order.append(node)