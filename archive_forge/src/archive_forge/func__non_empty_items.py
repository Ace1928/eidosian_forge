import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _non_empty_items(self):
    return [(k, vs) for k, vs in sorted(self.items()) if vs]