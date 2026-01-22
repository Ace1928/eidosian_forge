import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDeleteDirectoryThenAddFile(TestCaseForGenericProcessor):
    """Test deleting a directory then adding a file in the same commit."""

    def file_command_iter(self, paths, dir, new_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                for i, path in enumerate(paths):
                    yield commands.FileModifyCommand(path, kind_to_mode(kind, False), None, b'aaa%d' % i)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileDeleteCommand(dir)
                yield commands.FileModifyCommand(new_path, kind_to_mode(kind, False), None, b'bbb')
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_delete_dir_then_add_file(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d']
        dir = b'a/b'
        new_path = b'a/b/z'
        handler.process(self.file_command_iter(paths, dir, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',)], expected_added=[(b'a/b',), (b'a/b/z',)])
        self.assertContent(branch, revtree2, new_path, b'bbb')

    def test_delete_dir_then_add_symlink(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d']
        dir = b'a/b'
        new_path = b'a/b/z'
        handler.process(self.file_command_iter(paths, dir, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',)], expected_added=[(b'a/b',), (b'a/b/z',)])
        self.assertSymlinkTarget(branch, revtree2, new_path, 'bbb')