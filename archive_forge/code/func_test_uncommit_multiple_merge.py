import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
def test_uncommit_multiple_merge(self):
    wt = self.create_simple_tree()
    tree2 = wt.controldir.sprout('tree2').open_workingtree()
    tree2.commit('unchanged', rev_id=b'b3')
    tree3 = wt.controldir.sprout('tree3').open_workingtree()
    tree3.commit('unchanged', rev_id=b'c3')
    wt.merge_from_branch(tree2.branch)
    wt.commit('merge b3', rev_id=b'a3')
    wt.merge_from_branch(tree3.branch)
    wt.commit('merge c3', rev_id=b'a4')
    self.assertEqual([b'a4'], wt.get_parent_ids())
    os.chdir('tree')
    out, err = self.run_bzr('uncommit --force -r 2')
    self.assertEqual([b'a2', b'b3', b'c3'], wt.get_parent_ids())