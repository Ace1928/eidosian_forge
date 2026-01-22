from breezy import osutils, tests
from breezy.git.branch import GitBranch
from breezy.mutabletree import MutableTree
from breezy.tests import TestSkipped, features, per_tree
from breezy.transform import PreviewTree
class TestTreeWithoutSymlinks(per_tree.TestCaseWithTree):

    def setUp(self):
        super().setUp()
        self.branch = self.make_branch('a')
        mem_tree = self.branch.create_memorytree()
        with mem_tree.lock_write():
            mem_tree._file_transport.symlink('source', 'symlink')
            mem_tree.add(['', 'symlink'])
            rev1 = mem_tree.commit('rev1')
        self.assertPathDoesNotExist('a/symlink')

    def test_clone_skips_symlinks(self):
        if isinstance(self.branch, (GitBranch,)):
            raise TestSkipped('git trees do not honor osutils.supports_symlinks yet')
        self.overrideAttr(osutils, 'supports_symlinks', lambda p: False)
        result_dir = self.branch.controldir.sprout('b')
        result_tree = result_dir.open_workingtree()
        self.assertFalse(result_tree.supports_symlinks())
        self.assertPathDoesNotExist('b/symlink')
        basis_tree = self.branch.basis_tree()
        self.assertTrue(basis_tree.has_filename('symlink'))
        with result_tree.lock_read():
            self.assertEqual([('symlink', 'symlink')], [c.path for c in result_tree.iter_changes(basis_tree)])