import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_post_dominators(self):
    dummy_exit = object()
    self._exit_points.add(dummy_exit)
    for loop in self._loops.values():
        if not loop.exits:
            for b in loop.body:
                self._add_edge(b, dummy_exit)
    pdoms = self._find_dominators_internal(post=True)
    del pdoms[dummy_exit]
    for doms in pdoms.values():
        doms.discard(dummy_exit)
    self._remove_node_edges(dummy_exit)
    self._exit_points.remove(dummy_exit)
    return pdoms