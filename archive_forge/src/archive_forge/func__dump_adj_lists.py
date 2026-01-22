import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _dump_adj_lists(self, file):
    adj_lists = dict(((src, sorted(list(dests))) for src, dests in self._succs.items()))
    import pprint
    pprint.pprint(adj_lists, stream=file)