import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackRenameNew(TestCaseForGenericProcessor):
    """Test rename of a newly added file."""

    def get_command_iter(self, old_path, new_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(old_path, kind_to_mode(kind, False), None, b'aaa')
                yield commands.FileRenameCommand(old_path, new_path)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)
        return command_list

    def test_rename_new_file_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(new_path,)])
        self.assertRevisionRoot(revtree1, new_path)

    def test_rename_new_symlink_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(new_path,)])
        self.assertRevisionRoot(revtree1, new_path)

    def test_rename_new_file_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'a/b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (new_path,)])

    def test_rename_new_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'a/a'
        new_path = b'a/b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (new_path,)])