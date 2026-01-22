import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _eliminate_dead_blocks(self):
    """
        Eliminate all blocks not reachable from the entry point, and
        stash them into self._dead_nodes.
        """
    live = set()
    for node in self._dfs():
        live.add(node)
    self._dead_nodes = self._nodes - live
    self._nodes = live
    for dead in self._dead_nodes:
        self._remove_node_edges(dead)