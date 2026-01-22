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
def add_pyapi(self, pyapi, item, do_resize=True):
    """A version of .add for use inside functions following Python calling
        convention.
        """
    context = self._context
    builder = self._builder
    payload = self.payload
    h = self._pyapi_get_hash_value(pyapi, context, builder, item)
    self._add_key(payload, item, h, do_resize)