import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackRename(TestCaseForGenericProcessor):

    def get_command_iter(self, old_path, new_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(old_path, kind_to_mode(kind, False), None, b'aaa')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileRenameCommand(old_path, new_path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_rename_file_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)])
        self.assertRevisionRoot(revtree1, old_path)
        self.assertRevisionRoot(revtree2, new_path)

    def test_rename_symlink_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)])
        self.assertRevisionRoot(revtree1, old_path)
        self.assertRevisionRoot(revtree2, new_path)

    def test_rename_file_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'a/b'
        handler.process(self.get_command_iter(old_path, new_path))
        self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)])

    def test_rename_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'a/b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)])

    def test_rename_file_to_new_dir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'b/a'
        handler.process(self.get_command_iter(old_path, new_path))
        self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)], expected_added=[(b'b',)], expected_removed=[(b'a',)])

    def test_rename_symlink_to_new_dir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'b/a'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)], expected_added=[(b'b',)], expected_removed=[(b'a',)])