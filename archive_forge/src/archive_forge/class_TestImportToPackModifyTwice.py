import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackModifyTwice(TestCaseForGenericProcessor):
    """This tests when the same file is modified twice in the one commit.

    Note: hg-fast-export produces data like this on occasions.
    """

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
                yield commands.FileModifyCommand(path, kind_to_mode(to_kind, to_executable), None, to_content)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)
        return command_list

    def test_modify_file_twice_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertRevisionRoot(revtree1, path)