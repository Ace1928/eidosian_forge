import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_todo_nothing(self):
    self.run_bzr_error(['brz: ERROR: No rebase in progress'], ['rebase-todo'])