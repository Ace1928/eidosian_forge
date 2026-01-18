import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_subdir_commit(self):
    """Test committing a subdirectory, and committing a directory."""
    tree = self.make_branch_and_tree('.')
    b = tree.branch
    self.build_tree(['a/', 'b/'])

    def set_contents(contents):
        self.build_tree_contents([('a/one', contents), ('b/two', contents), ('top', contents)])
    set_contents(b'old contents')
    tree.smart_add(['.'])
    tree.commit('first revision')
    set_contents(b'new contents')
    mutter('start selective subdir commit')
    self.run_bzr(['commit', 'a', '-m', 'commit a only'])
    new = b.repository.revision_tree(b.get_rev_id(2))
    new.lock_read()

    def get_text_by_path(tree, path):
        return tree.get_file_text(path)
    self.assertEqual(get_text_by_path(new, 'b/two'), b'old contents')
    self.assertEqual(get_text_by_path(new, 'top'), b'old contents')
    self.assertEqual(get_text_by_path(new, 'a/one'), b'new contents')
    new.unlock()
    self.run_bzr(['commit', '.', '-m', 'commit subdir only', '--unchanged'], working_dir='a')
    v3 = b.repository.revision_tree(b.get_rev_id(3))
    v3.lock_read()
    self.assertEqual(get_text_by_path(v3, 'b/two'), b'old contents')
    self.assertEqual(get_text_by_path(v3, 'top'), b'old contents')
    self.assertEqual(get_text_by_path(v3, 'a/one'), b'new contents')
    v3.unlock()
    self.run_bzr(['commit', '-m', 'commit whole tree from subdir'], working_dir='a')
    v4 = b.repository.revision_tree(b.get_rev_id(4))
    v4.lock_read()
    self.assertEqual(get_text_by_path(v4, 'b/two'), b'new contents')
    self.assertEqual(get_text_by_path(v4, 'top'), b'new contents')
    v4.unlock()