import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_sync_delete(self):
    """ Test that deleting a synchronized trait works.
        """
    a = A()
    b = B()
    a.sync_trait('t', b)
    a.t = 5
    del a
    try:
        b.t = 7
    except Exception:
        self.fail('Unexpected exception while setting sync trait.')