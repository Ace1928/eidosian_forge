import inspect
import typing as py_typing
from numba.core.typing.typeof import typeof
from numba.core import errors, types
def try_infer(self, py_type):
    """
        Try to determine the numba type of a given python type.
        We first consider the lookup dictionary.  If py_type is not there, we
        iterate through the registered functions until one returns a numba type.
        If type inference fails, return None.
        """
    result = self.lookup.get(py_type, None)
    for func in self.functions:
        if result is not None:
            break
        result = func(py_type)
    if result is not None and (not isinstance(result, types.Type)):
        raise errors.TypingError(f'as_numba_type should return a numba type, got {result}')
    return result