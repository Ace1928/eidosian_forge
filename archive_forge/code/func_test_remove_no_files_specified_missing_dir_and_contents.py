import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_no_files_specified_missing_dir_and_contents(self):
    tree = self._make_tree_and_add(['foo', 'dir/', 'dir/missing/', 'dir/missing/child'])
    self.get_transport('.').delete_tree('dir/missing')
    out, err = self.run_bzr(['rm'])
    self.assertEqual('', out)
    self.assertEqual('removed dir/missing/child\nremoved dir/missing\n', err)
    self.assertInWorkingTree('foo', tree=tree)
    self.assertPathExists('foo')
    self.assertInWorkingTree('dir', tree=tree)
    self.assertPathExists('dir')
    self.assertNotInWorkingTree('dir/missing', tree=tree)
    self.assertNotInWorkingTree('dir/missing/child', tree=tree)