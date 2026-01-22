import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def _require_nrt(self):
    if not self._enabled:
        raise errors.NumbaRuntimeError('NRT required but not enabled')