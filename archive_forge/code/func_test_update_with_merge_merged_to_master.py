import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_with_merge_merged_to_master(self):
    master = self.make_branch_and_tree('master')
    self.build_tree(['master/file'])
    master.add(['file'])
    master.commit('one', rev_id=b'm1')
    self.build_tree(['checkout1/'])
    checkout_dir = bzrdir.BzrDirMetaFormat1().initialize('checkout1')
    checkout_dir.set_branch_reference(master.branch)
    checkout1 = checkout_dir.create_workingtree(b'm1')
    other = master.controldir.sprout('other').open_workingtree()
    self.build_tree(['other/file2'])
    other.add(['file2'])
    other.commit('other2', rev_id=b'o2')
    checkout1.merge_from_branch(other.branch)
    self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])
    master.merge_from_branch(other.branch)
    master.commit('f3', rev_id=b'm2')
    out, err = self.run_bzr('update', working_dir='checkout1')
    self.assertEqual('', out)
    self.assertEqualDiff('All changes applied successfully.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'master'), err)
    self.assertEqual([], checkout1.get_parent_ids()[1:])