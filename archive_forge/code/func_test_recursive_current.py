from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport
def test_recursive_current(self):
    self.run_bzr('init .')
    self.assertEqual('.\n', self.run_bzr('branches --recursive')[0])