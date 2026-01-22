import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def assertCompareNoRelation(self, k1, k2, mismatched_types=False):
    """Run the comparison operators, make sure they do something.

        However, we don't actually care what comes first or second. This is
        stuff like cross-class comparisons. We don't want to segfault/raise an
        exception, but we don't care about the sort order.
        """
    self.assertFalse(k1 == k2)
    self.assertTrue(k1 != k2)
    if not self.check_strict_compare(k1, k2, mismatched_types):
        k1 >= k2
        k1 > k2
        k1 <= k2
        k1 < k2