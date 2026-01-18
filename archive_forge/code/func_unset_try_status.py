from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def unset_try_status(self, builder):
    try_state_ptr = self._get_try_state(builder)
    old = builder.load(try_state_ptr)
    new = builder.sub(old, old.type(1))
    builder.store(new, try_state_ptr)
    excinfoptr = self._get_excinfo_argument(builder.function)
    null = cgutils.get_null_value(excinfoptr.type.pointee)
    builder.store(null, excinfoptr)