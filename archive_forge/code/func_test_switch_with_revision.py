import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_with_revision(self):
    """Test switch when a revision is given."""
    tree = self.make_branch_and_tree('branch-1')
    self.build_tree(['branch-1/file-1'])
    tree.add('file-1')
    tree.commit(rev_id=b'rev1', message='rev1')
    self.build_tree(['branch-1/file-2'])
    tree.add('file-2')
    tree.commit(rev_id=b'rev2', message='rev2')
    checkout = tree.branch.create_checkout('checkout', lightweight=self.lightweight)
    switch.switch(checkout.controldir, tree.branch, revision_id=b'rev1')
    self.assertPathExists('checkout/file-1')
    self.assertPathDoesNotExist('checkout/file-2')