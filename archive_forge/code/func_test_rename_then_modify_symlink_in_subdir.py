import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_then_modify_symlink_in_subdir(self):
    handler, branch = self.get_handler()
    old_path = b'd/a'
    new_path = b'd/b'
    handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd',), (old_path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)])
    self.assertSymlinkTarget(branch, revtree1, old_path, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, new_path, 'bbb')