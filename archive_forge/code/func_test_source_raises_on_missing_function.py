from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_source_raises_on_missing_function():
    f = Dispatcher('f')
    assert raises(TypeError, lambda: f.source(1))