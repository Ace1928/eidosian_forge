import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogMultiple(TestLogWithLogCatcher):

    def prepare_tree(self):
        tree = self.make_branch_and_tree('parent')
        self.build_tree(['parent/file1', 'parent/file2', 'parent/dir1/', 'parent/dir1/file5', 'parent/dir1/dir2/', 'parent/dir1/dir2/file3', 'parent/file4'])
        tree.add('file1')
        tree.commit('add file1')
        tree.add('file2')
        tree.commit('add file2')
        tree.add(['dir1', 'dir1/dir2', 'dir1/dir2/file3'])
        tree.commit('add file3')
        tree.add('file4')
        tree.commit('add file4')
        tree.add('dir1/file5')
        tree.commit('add file5')
        child_tree = tree.controldir.sprout('child').open_workingtree()
        self.build_tree_contents([('child/file2', b'hello')])
        child_tree.commit(message='branch 1')
        tree.merge_from_branch(child_tree.branch)
        tree.commit(message='merge child branch')
        os.chdir('parent')

    def test_log_files(self):
        """The log for multiple file should only list revs for those files"""
        self.prepare_tree()
        self.assertLogRevnos(['file1', 'file2', 'dir1/dir2/file3'], ['6', '5.1.1', '3', '2', '1'])

    def test_log_directory(self):
        """The log for a directory should show all nested files."""
        self.prepare_tree()
        self.assertLogRevnos(['dir1'], ['5', '3'])

    def test_log_nested_directory(self):
        """The log for a directory should show all nested files."""
        self.prepare_tree()
        self.assertLogRevnos(['dir1/dir2'], ['3'])

    def test_log_in_nested_directory(self):
        """The log for a directory should show all nested files."""
        self.prepare_tree()
        os.chdir('dir1')
        self.assertLogRevnos(['.'], ['5', '3'])

    def test_log_files_and_directories(self):
        """Logging files and directories together should be fine."""
        self.prepare_tree()
        self.assertLogRevnos(['file4', 'dir1/dir2'], ['4', '3'])

    def test_log_files_and_dirs_in_nested_directory(self):
        """The log for a directory should show all nested files."""
        self.prepare_tree()
        os.chdir('dir1')
        self.assertLogRevnos(['dir2', 'file5'], ['5', '3'])