from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def _get_try_state(self, builder):
    try:
        return builder.__eh_try_state
    except AttributeError:
        ptr = cgutils.alloca_once(builder, cgutils.intp_t, name='try_state', zfill=True)
        builder.__eh_try_state = ptr
        return ptr