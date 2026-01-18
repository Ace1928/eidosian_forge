import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest
def test_overload_float_error_msg(self):
    mock_float = gen_mock_float()
    self._add_float_overload(mock_float)

    @njit
    def foo(x):
        return mock_float(x)
    with self.assertRaises(TypingError) as raises:
        foo(1j)
    self.assertIn('cannot type float(complex128)', str(raises.exception))