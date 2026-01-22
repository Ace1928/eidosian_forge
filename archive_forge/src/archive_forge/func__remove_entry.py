import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
def _remove_entry(self, payload, entry, do_resize=True, do_decref=True):
    entry.hash = ir.Constant(entry.hash.type, DELETED)
    if do_decref:
        self.decref_value(entry.key)
    used = payload.used
    one = ir.Constant(used.type, 1)
    used = payload.used = self._builder.sub(used, one)
    if do_resize:
        self.downsize(used)
    self.set_dirty(True)