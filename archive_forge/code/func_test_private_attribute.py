import platform
import time
import unittest
import pytest
from monty.functools import (
def test_private_attribute(self):
    called = []

    class Foo:

        @lazy_property
        def __foo(self):
            called.append('foo')
            return 1

        def get_foo(self):
            return self.__foo
    f = Foo()
    assert f.get_foo() == 1
    assert f.get_foo() == 1
    assert f.get_foo() == 1
    assert len(called) == 1