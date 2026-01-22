import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
@functools.cached_property
def _domtree(self):
    return self._find_dominator_tree()