import os
from breezy import bedding, tests, workingtree
def _make_simple_branch(self, relpath='.'):
    wt = self.make_branch_and_tree(relpath)
    wt.commit('first revision')
    wt.commit('second revision')
    return wt