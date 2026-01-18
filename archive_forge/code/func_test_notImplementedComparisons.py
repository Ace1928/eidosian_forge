import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_notImplementedComparisons(self) -> None:
    """
        Comparing a L{Version} to some other object type results in
        C{NotImplemented}.
        """
    va = Version('dummy', 1, 0, 0)
    vb = ('dummy', 1, 0, 0)
    result = va.__cmp__(vb)
    self.assertEqual(result, NotImplemented)