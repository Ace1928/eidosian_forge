import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert_chatter(self):
    self._prepare_rename_mod_tree()
    chatter = self.run_bzr('revert')[1]
    self.assertEqualDiff('R   a/g => f/g\nR   h => f/h\nR   j/ => f/\nR   j/b => a/b\nR   j/d/ => a/d/\nR   j/e => a/d/e\n', chatter)