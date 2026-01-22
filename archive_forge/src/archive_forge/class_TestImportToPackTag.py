import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackTag(TestCaseForGenericProcessor):

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
                yield commands.FileModifyCommand(path, kind_to_mode(to_kind, to_executable), None, to_content)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b'head', [], files_two)
            yield commands.TagCommand(b'tag1', b':1', committer, b'tag 1')
            yield commands.TagCommand(b'tag2', b'head', committer, b'tag 2')
        return command_list

    def test_tag(self):
        handler, branch = self.get_handler()
        path = b'a'
        raise tests.KnownFailure('non-mark committish not yet supported- bug #410249')
        handler.process(self.file_command_iter(path))