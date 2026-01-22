import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
class FooTest(unittest.SynchronousTestCase):

    def test_foo(self) -> None:
        pass

    def test_bar(self) -> None:
        pass