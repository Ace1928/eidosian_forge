import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackCopyNew(TestCaseForGenericProcessor):
    """Test copy of a newly added file."""

    def file_command_iter(self, src_path, dest_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(src_path, kind_to_mode(kind, False), None, b'aaa')
                yield commands.FileCopyCommand(src_path, dest_path)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)
        return command_list

    def test_copy_new_file_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(src_path,), (dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree1, dest_path, b'aaa')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree1, dest_path)

    def test_copy_new_file_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'a/b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (src_path,), (dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree1, dest_path, b'aaa')

    def test_copy_new_file_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'b/a'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (src_path,), (b'b',), (dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree1, dest_path, b'aaa')

    def test_copy_new_symlink_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(src_path,), (dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, dest_path, 'aaa')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree1, dest_path)

    def test_copy_new_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'a/b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (src_path,), (dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, dest_path, 'aaa')

    def test_copy_new_symlink_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'b/a'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (src_path,), (b'b',), (dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, dest_path, 'aaa')