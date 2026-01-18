import unittest
from traits.api import (
def test_union_unspecified_arguments(self):

    class TestClass(HasTraits):
        none = Union()
    TestClass(none=None)