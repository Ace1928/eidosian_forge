import inspect
import typing as py_typing
from numba.core.typing.typeof import typeof
from numba.core import errors, types
def _numba_type_infer(self, py_type):
    if isinstance(py_type, types.Type):
        return py_type