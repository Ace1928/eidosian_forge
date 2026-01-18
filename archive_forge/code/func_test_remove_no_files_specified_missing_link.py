import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_files_specified_missing_link(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self._make_tree_and_add(['foo'])
    os.symlink('foo', 'linkname')
    tree.add(['linkname'])
    os.unlink('linkname')
    out, err = self.run_bzr(['rm'])
    self.assertEqual('', out)
    self.assertEqual('removed linkname\n', err)
    self.assertInWorkingTree('foo', tree=tree)
    self.assertPathExists('foo')
    self.assertNotInWorkingTree('linkname', tree=tree)