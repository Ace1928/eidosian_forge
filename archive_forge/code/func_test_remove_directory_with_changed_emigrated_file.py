from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_directory_with_changed_emigrated_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('somedir/',), (b'somedir/file', b'contents')])
    tree.add(['somedir', 'somedir/file'])
    tree.commit(message='first')
    self.build_tree_contents([('somedir/file', b'changed')])
    tree.rename_one('somedir/file', 'moved-file')
    tree.remove('somedir', keep_files=False)
    self.assertNotInWorkingTree('somedir')
    self.assertPathDoesNotExist('somedir')
    self.assertInWorkingTree('moved-file')
    self.assertPathExists('moved-file')