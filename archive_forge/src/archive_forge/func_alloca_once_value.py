import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def alloca_once_value(builder, value, name='', zfill=False):
    """
    Like alloca_once(), but passing a *value* instead of a type.  The
    type is inferred and the allocated slot is also initialized with the
    given value.
    """
    storage = alloca_once(builder, value.type, zfill=zfill)
    builder.store(value, storage)
    return storage