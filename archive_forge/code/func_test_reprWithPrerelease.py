import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_reprWithPrerelease(self) -> None:
    """
        Calling C{repr} on a version with a prerelease returns a human-readable
        string representation of the version including the prerelease.
        """
    self.assertEqual(repr(Version('dummy', 1, 2, 3, prerelease=4)), "Version('dummy', 1, 2, 3, release_candidate=4)")