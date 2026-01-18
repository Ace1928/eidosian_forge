import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_files_specified_missing_file(self):
    tree = self._make_tree_and_add(['foo', 'bar'])
    os.unlink('bar')
    out, err = self.run_bzr(['rm'])
    self.assertEqual('', out)
    self.assertEqual('removed bar\n', err)
    self.assertInWorkingTree('foo', tree=tree)
    self.assertPathExists('foo')
    self.assertNotInWorkingTree('bar', tree=tree)