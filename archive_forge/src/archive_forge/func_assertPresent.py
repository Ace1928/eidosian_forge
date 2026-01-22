from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def assertPresent(self, expected, vf, keys):
    """Check which of the supplied keys are present."""
    parent_map = vf.get_parent_map(keys)
    self.assertEqual(sorted(expected), sorted(parent_map))