from breezy import controldir
from breezy.tests import TestCaseWithTransport
class TestRemoveBranch(TestCaseWithTransport):

    def example_tree(self, path='.', format=None):
        tree = self.make_branch_and_tree(path, format=format)
        self.build_tree_contents([(path + '/hello', b'foo')])
        tree.add('hello')
        tree.commit(message='setup')
        self.build_tree_contents([(path + '/goodbye', b'baz')])
        tree.add('goodbye')
        tree.commit(message='setup')
        return tree

    def test_remove_local(self):
        tree = self.example_tree('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
        self.run_bzr('rmbranch --force a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())
        self.assertPathExists('a/hello')
        self.assertPathExists('a/goodbye')

    def test_no_branch(self):
        self.make_repository('a')
        self.run_bzr_error(['Not a branch'], 'rmbranch a')

    def test_no_tree(self):
        tree = self.example_tree('a')
        tree.controldir.destroy_workingtree()
        self.run_bzr('rmbranch', working_dir='a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())

    def test_no_arg(self):
        self.example_tree('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch a')
        self.run_bzr('rmbranch --force', working_dir='a')
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch())

    def test_remove_colo(self):
        tree = self.example_tree('a')
        tree.controldir.create_branch(name='otherbranch')
        self.assertTrue(tree.controldir.has_branch('otherbranch'))
        self.run_bzr('rmbranch %s,branch=otherbranch' % tree.controldir.user_url)
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch('otherbranch'))
        self.assertTrue(dir.has_branch())

    def test_remove_colo_directory(self):
        tree = self.example_tree('a')
        tree.controldir.create_branch(name='otherbranch')
        self.assertTrue(tree.controldir.has_branch('otherbranch'))
        self.run_bzr('rmbranch otherbranch -d %s' % tree.controldir.user_url)
        dir = controldir.ControlDir.open('a')
        self.assertFalse(dir.has_branch('otherbranch'))
        self.assertTrue(dir.has_branch())

    def test_remove_active_colo_branch(self):
        dir = self.make_repository('a').controldir
        branch = dir.create_branch('otherbranch')
        branch.create_checkout('a')
        self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch otherbranch -d %s' % branch.controldir.user_url)
        self.assertTrue(dir.has_branch('otherbranch'))
        self.run_bzr('rmbranch --force otherbranch -d %s' % branch.controldir.user_url)
        self.assertFalse(dir.has_branch('otherbranch'))