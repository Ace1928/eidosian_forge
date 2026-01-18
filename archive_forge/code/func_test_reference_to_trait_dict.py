import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
def test_reference_to_trait_dict(self):
    """ Does a HasTraits object refer to its __dict__ object?

            This test may point to why the previous one fails.  Even if it
            doesn't, the functionality is needed for detecting problems
            with memory in debug.memory_tracker
        """

    class Foo(HasTraits):
        child = Any
    foo = Foo()
    time.sleep(0.1)
    referrers = gc.get_referrers(foo.__dict__)
    self.assertTrue(len(referrers) > 0)
    self.assertTrue(foo in referrers)