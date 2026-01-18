import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
def test_simple_cycle_hastraits(self):
    """ Can the garbage collector clean up a cycle with traits objects?
        """

    class Foo(HasTraits):
        child = Any
    self._simple_cycle_helper(Foo)