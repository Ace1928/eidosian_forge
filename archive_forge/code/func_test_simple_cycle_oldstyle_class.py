import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
def test_simple_cycle_oldstyle_class(self):
    """ Can the garbage collector clean up a cycle with old style class?
        """

    class Foo:

        def __init__(self, child=None):
            self.child = child
    self._simple_cycle_helper(Foo)