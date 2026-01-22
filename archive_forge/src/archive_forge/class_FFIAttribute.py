from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
@registry.register_attr
class FFIAttribute(templates.AttributeTemplate):
    key = types.ffi

    def resolve_from_buffer(self, ffi):
        return types.BoundFunction(FFI_from_buffer, types.ffi)