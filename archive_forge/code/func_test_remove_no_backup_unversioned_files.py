import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_backup_unversioned_files(self):
    self.build_tree(files)
    tree = self.make_branch_and_tree('.')
    script.ScriptRunner().run_script(self, '\n        $ brz remove --no-backup a b/ b/c d/\n        2>deleted d\n        2>removed b/c (but kept a copy: b/c.~1~)\n        2>deleted b\n        2>deleted a\n        ')
    self.assertFilesDeleted(files)