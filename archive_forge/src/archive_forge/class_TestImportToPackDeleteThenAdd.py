import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDeleteThenAdd(TestCaseForGenericProcessor):
    """Test delete followed by an add. Merges can cause this."""

    def file_command_iter(self, path, kind='file', content=b'aaa', executable=False, to_kind=None, to_content=b'bbb', to_executable=None):
        if to_kind is None:
            to_kind = kind
        if to_executable is None:
            to_executable = executable

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path, kind_to_mode(kind, executable), None, content)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileDeleteCommand(path)
                yield commands.FileModifyCommand(path, kind_to_mode(to_kind, to_executable), None, to_content)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_delete_then_add_file_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)], expected_added=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertContent(branch, revtree2, path, b'bbb')
        self.assertRevisionRoot(revtree1, path)
        self.assertRevisionRoot(revtree2, path)

    def test_delete_then_add_file_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)], expected_added=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertContent(branch, revtree2, path, b'bbb')

    def test_delete_then_add_symlink_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)], expected_added=[(path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, path, 'bbb')
        self.assertRevisionRoot(revtree1, path)
        self.assertRevisionRoot(revtree2, path)

    def test_delete_then_add_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)], expected_added=[(path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, path, 'bbb')