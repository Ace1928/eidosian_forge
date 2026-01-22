import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def _check_null_result(func):

    @functools.wraps(func)
    def wrap(self, builder, *args, **kwargs):
        memptr = func(self, builder, *args, **kwargs)
        msg = 'Allocation failed (probably too large).'
        cgutils.guard_memory_error(self._context, builder, memptr, msg=msg)
        return memptr
    return wrap