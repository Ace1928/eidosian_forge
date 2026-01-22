from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _make_canonical_test_tree(self, commit=True):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/dir/', 'tree/dir/file'])
    work_tree.add(['dir', 'dir/file'])
    if commit:
        work_tree.commit('commit 1')
    return work_tree