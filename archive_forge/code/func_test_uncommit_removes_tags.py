import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_removes_tags(self):
    tree = self.make_branch_and_tree('tree')
    revid = tree.commit('message')
    tree.branch.tags.set_tag('atag', revid)
    out, err = self.run_bzr('uncommit --force tree')
    self.assertEqual({}, tree.branch.tags.get_tag_dict())