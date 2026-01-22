import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def assertRefcount(self, count, obj):
    """Assert that the refcount for obj is what we expect.

        Note that this automatically adjusts for the fact that calling
        assertRefcount actually creates a new pointer, as does calling
        sys.getrefcount. So pass the expected value *before* the call.
        """
    if sys.version_info < (3, 11):
        self.assertEqual(count, sys.getrefcount(obj) - 3)
    else:
        self.assertEqual(count, sys.getrefcount(obj) - 2)