import platform
import time
import unittest
import pytest
from monty.functools import (
def test_reserved_attribute(self):
    called = []

    class Foo:

        @lazy_property
        def __foo__(self):
            called.append('foo')
            return 1
    f = Foo()
    assert f.__foo__ == 1
    assert f.__foo__ == 1
    assert f.__foo__ == 1
    assert len(called) == 1