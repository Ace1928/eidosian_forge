import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackFileKinds(TestCaseForGenericProcessor):

    def get_command_iter(self, path, kind, content):

        def command_list():
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path, kind_to_mode(kind, False), None, content)
            yield commands.CommitCommand(b'head', b'1', None, committer, b'commit 1', None, [], files_one)
        return command_list

    def test_import_plainfile(self):
        handler, branch = self.get_handler()
        handler.process(self.get_command_iter(b'foo', 'file', b'aaa'))

    def test_import_symlink(self):
        handler, branch = self.get_handler()
        handler.process(self.get_command_iter(b'foo', 'symlink', b'bar'))