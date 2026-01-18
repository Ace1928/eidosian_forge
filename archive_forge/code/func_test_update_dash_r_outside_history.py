import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_dash_r_outside_history(self):
    """Ensure that we can update -r to dotted revisions.
        """
    master = self.make_branch_and_tree('master')
    self.build_tree(['master/file1'])
    master.add(['file1'])
    master.commit('one', rev_id=b'm1')
    other = master.controldir.sprout('other').open_workingtree()
    self.build_tree(['other/file2', 'other/file3'])
    other.add(['file2'])
    other.commit('other2', rev_id=b'o2')
    other.add(['file3'])
    other.commit('other3', rev_id=b'o3')
    os.chdir('master')
    self.run_bzr('merge ../other')
    master.commit('merge', rev_id=b'merge')
    out, err = self.run_bzr('update -r revid:o2')
    self.assertContainsRe(err, '-D\\s+file3')
    self.assertContainsRe(err, 'All changes applied successfully\\.')
    self.assertContainsRe(err, 'Updated to revision 1.1.1 of branch .*')
    out, err = self.run_bzr('update')
    self.assertContainsRe(err, '\\+N\\s+file3')
    self.assertContainsRe(err, 'All changes applied successfully\\.')
    self.assertContainsRe(err, 'Updated to revision 2 of branch .*')