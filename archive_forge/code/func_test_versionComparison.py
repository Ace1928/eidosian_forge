import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_versionComparison(self) -> None:
    """
        Versions can be compared for equality and order.
        """
    va = Version('dummy', 1, 0, 0)
    vb = Version('dummy', 0, 1, 0)
    self.assertTrue(va > vb)
    self.assertTrue(vb < va)
    self.assertTrue(va >= vb)
    self.assertTrue(vb <= va)
    self.assertTrue(va != vb)
    self.assertTrue(vb == Version('dummy', 0, 1, 0))
    self.assertTrue(vb == vb)