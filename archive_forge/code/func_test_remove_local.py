from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_remove_local(self):
    tree = self.example_tree('a')
    self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
    self.run_bzr('rmbranch --force a')
    dir = controldir.ControlDir.open('a')
    self.assertFalse(dir.has_branch())
    self.assertPathExists('a/hello')
    self.assertPathExists('a/goodbye')