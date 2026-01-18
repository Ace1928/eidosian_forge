import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_revision_info_tree(self):
    wt = self.make_branch_and_tree('branch')
    wt.commit('Commit one', rev_id=b'a@r-0-1')
    wt.branch.create_checkout('checkout', lightweight=True)
    wt.commit('Commit two', rev_id=b'a@r-0-2')
    self.check_output('2 a@r-0-2\n', 'revision-info -d checkout')
    self.check_output('1 a@r-0-1\n', 'revision-info --tree -d checkout')