import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_files_specified(self):
    tree = self._make_tree_and_add(['foo'])
    out, err = self.run_bzr(['rm'])
    self.assertEqual('', err)
    self.assertEqual('', out)
    self.assertInWorkingTree('foo', tree=tree)
    self.assertPathExists('foo')