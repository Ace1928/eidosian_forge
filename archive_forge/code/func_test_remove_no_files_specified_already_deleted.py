import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_files_specified_already_deleted(self):
    tree = self._make_tree_and_add(['foo', 'bar'])
    tree.commit('save foo and bar')
    os.unlink('bar')
    self.run_bzr(['rm'])
    self.assertFalse(tree.is_versioned('bar'))
    out, err = self.run_bzr(['rm'])
    self.assertEqual('', out)
    self.assertEqual('', err)