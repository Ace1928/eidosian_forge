import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_readonly_lightweight_update(self):
    """Update a light checkout of a readonly branch"""
    tree = self.make_branch_and_tree('branch')
    readonly_branch = branch.Branch.open(self.get_readonly_url('branch'))
    checkout = readonly_branch.create_checkout('checkout', lightweight=True)
    tree.commit('empty commit')
    self.run_bzr('update checkout')