import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def _setup_uncommitted(self, same_revision=False):
    tree = self._setup_tree()
    to_branch = tree.controldir.sprout('branch-2').open_branch()
    self.build_tree(['branch-1/file-2'])
    if not same_revision:
        tree.add('file-2')
        tree.remove('file-1')
        tree.commit('rev2')
    checkout = tree.branch.create_checkout('checkout', lightweight=self.lightweight)
    self.build_tree(['checkout/file-3'])
    checkout.add('file-3')
    return (checkout, to_branch)