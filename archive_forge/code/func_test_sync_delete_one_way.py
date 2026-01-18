import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_sync_delete_one_way(self):
    """ Test that deleting a one-way synchronized trait works.

        (Regression test for #131).

        """
    a = A()
    b = B()
    a.sync_trait('t', b, mutual=False)
    del b
    try:
        a.t = 42
    except Exception:
        self.fail('Unexpected exception while setting sync trait.')