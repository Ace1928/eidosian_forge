import platform
import time
import unittest
import pytest
from monty.functools import (
class TestTryOrReturn:

    def test_decorator(self):

        class A:

            @return_if_raise(ValueError, 'hello')
            def return_one(self):
                return 1

            @return_if_raise(ValueError, 'hello')
            def return_hello(self):
                raise ValueError()

            @return_if_raise(KeyError, 'hello')
            def reraise_value_error(self):
                raise ValueError()

            @return_if_raise([KeyError, ValueError], 'hello')
            def catch_exc_list(self):
                import random
                if random.randint(0, 1) == 0:
                    raise ValueError()
                else:
                    raise KeyError()

            @return_none_if_raise(TypeError)
            def return_none(self):
                raise TypeError()
        a = A()
        assert a.return_one() == 1
        assert a.return_hello() == 'hello'
        with pytest.raises(ValueError):
            a.reraise_value_error()
        assert a.catch_exc_list() == 'hello'
        assert a.return_none() is None