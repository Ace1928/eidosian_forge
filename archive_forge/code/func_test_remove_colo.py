from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_remove_colo(self):
    tree = self.example_tree('a')
    tree.controldir.create_branch(name='otherbranch')
    self.assertTrue(tree.controldir.has_branch('otherbranch'))
    self.run_bzr('rmbranch %s,branch=otherbranch' % tree.controldir.user_url)
    dir = controldir.ControlDir.open('a')
    self.assertFalse(dir.has_branch('otherbranch'))
    self.assertTrue(dir.has_branch())