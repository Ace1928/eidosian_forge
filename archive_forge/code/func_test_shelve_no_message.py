import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_shelve_no_message(self):
    tree = self.make_branch_and_tree('.')
    creator = self.make_creator(tree)
    shelf_id = tree.get_shelf_manager().shelve_changes(creator)
    out, err = self.run_bzr('shelve --list', retcode=1)
    self.assertEqual('', err)
    self.assertEqual('  1: <no message>\n', out)