from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_raise_error_on_non_class():
    f = Dispatcher('f')
    assert raises(TypeError, lambda: f.add((1,), inc))