from numba.core import types, typing
from numba.core.cgutils import unpack_tuple
from numba.core.extending import intrinsic
from numba.core.imputils import impl_ret_new_ref
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
Convert *array* into a tuple of *length*

    Returns ``UniTuple(array.dtype, length)``

    ** Warning **
    - No boundchecking.
      If *length* is longer than *array.size*, the behavior is undefined.
    