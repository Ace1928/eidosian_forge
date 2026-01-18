import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_subclasses(self):

    class MockSub(Mock):
        pass
    one = MockSub()
    self.assertTrue(issubclass(type(one.one), MockSub))

    class MagicSub(MagicMock):
        pass
    two = MagicSub()
    self.assertTrue(issubclass(type(two.two), MagicSub))