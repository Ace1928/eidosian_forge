import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
class PyunitTest(pyunit.TestCase):

    def test_foo(self) -> None:
        pass

    def test_bar(self) -> None:
        pass