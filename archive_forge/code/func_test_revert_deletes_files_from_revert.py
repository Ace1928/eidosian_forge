import os
from breezy import merge, tests, transform, workingtree
def test_revert_deletes_files_from_revert(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    tree.add('file')
    rev1 = tree.commit('added file')
    with tree.lock_read():
        file_sha = tree.get_file_sha1('file')
    os.unlink('file')
    tree.commit('removed file')
    self.assertPathDoesNotExist('file')
    tree.revert(old_tree=tree.branch.repository.revision_tree(rev1))
    self.assertEqual({'file': file_sha}, tree.merge_modified())
    self.assertPathExists('file')
    tree.revert()
    self.assertPathDoesNotExist('file')
    self.assertEqual({}, tree.merge_modified())