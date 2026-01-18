import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_nonascii(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('ሴ message')
    out, err = self.run_bzr('uncommit --force tree', encoding='ascii')
    self.assertContainsRe(out, '\\? message')