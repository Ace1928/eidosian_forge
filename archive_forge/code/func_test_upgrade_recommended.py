from breezy import errors
from breezy.tests.per_controldir import TestCaseWithControlDir
def test_upgrade_recommended(self):
    self.assertIsInstance(self.bzrdir_format.upgrade_recommended, bool)