import inspect
import typing as py_typing
from numba.core.typing.typeof import typeof
from numba.core import errors, types

        Try to determine the numba type of a given python type.
        We first consider the lookup dictionary.  If py_type is not there, we
        iterate through the registered functions until one returns a numba type.
        If type inference fails, return None.
        