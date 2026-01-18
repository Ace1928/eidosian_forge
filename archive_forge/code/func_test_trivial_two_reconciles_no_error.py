from breezy.tests.per_repository import TestCaseWithRepository
def test_trivial_two_reconciles_no_error(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('first post')
    tree.branch.repository.reconcile(thorough=True)
    tree.branch.repository.reconcile(thorough=True)