import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
def _simple_cycle_helper(self, foo_class):
    """ Can the garbage collector clean up a cycle with traits objects?
        """
    first = foo_class()
    second = foo_class(child=first)
    first.child = second
    foo_ids = [id(first), id(second)]
    del first, second
    gc.collect()
    all_ids = [id(obj) for obj in gc.get_objects()]
    for foo_id in foo_ids:
        self.assertTrue(foo_id not in all_ids)