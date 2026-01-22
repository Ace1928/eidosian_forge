import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackCopyModified(TestCaseForGenericProcessor):
    """Test copy of file/symlink already modified in this commit."""

    def file_command_iter(self, src_path, dest_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(src_path, kind_to_mode(kind, False), None, b'aaa')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileModifyCommand(src_path, kind_to_mode(kind, False), None, b'bbb')
                yield commands.FileCopyCommand(src_path, dest_path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_copy_of_modified_file_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'bbb')
        self.assertContent(branch, revtree2, dest_path, b'bbb')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree2, dest_path)

    def test_copy_of_modified_file_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'd/a'
        dest_path = b'd/b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'bbb')
        self.assertContent(branch, revtree2, dest_path, b'bbb')

    def test_copy_of_modified_file_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'd1/a'
        dest_path = b'd2/a'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(b'd2',), (dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'bbb')
        self.assertContent(branch, revtree2, dest_path, b'bbb')

    def test_copy_of_modified_symlink_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'bbb')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree2, dest_path)

    def test_copy_of_modified_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'd/a'
        dest_path = b'd/b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'bbb')

    def test_copy_of_modified_symlink_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'd1/a'
        dest_path = b'd2/a'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(src_path,)], expected_added=[(b'd2',), (dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'bbb')