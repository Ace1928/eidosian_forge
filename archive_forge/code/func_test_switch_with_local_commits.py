import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_with_local_commits(self):
    """Test switch complains about local commits unless --force given."""
    tree = self._setup_tree()
    to_branch = tree.controldir.sprout('branch-2').open_branch()
    self.build_tree(['branch-1/file-2'])
    tree.add('file-2')
    tree.remove('file-1')
    tree.commit('rev2')
    checkout = tree.branch.create_checkout('checkout')
    self.build_tree(['checkout/file-3'])
    checkout.add('file-3')
    checkout.commit(message='local only commit', local=True)
    self.build_tree(['checkout/file-4'])
    err = self.assertRaises(errors.CommandError, switch.switch, checkout.controldir, to_branch)
    self.assertContainsRe(str(err), 'Cannot switch as local commits found in the checkout.')
    self.assertPathDoesNotExist('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    switch.switch(checkout.controldir, to_branch, force=True)
    self.assertPathExists('checkout/file-1')
    self.assertPathDoesNotExist('checkout/file-2')
    self.assertPathDoesNotExist('checkout/file-3')
    self.assertPathExists('checkout/file-4')
    self.assertEqual(to_branch.last_revision_info(), checkout.branch.last_revision_info())