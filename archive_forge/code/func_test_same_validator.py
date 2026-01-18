from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_same_validator(self):
    tree = self.make_branch_and_tree('tree')
    revid = tree.commit('empty post')
    revtree = tree.branch.repository.revision_tree(tree.branch.last_revision())
    tree.basis_tree()
    revtree.lock_read()
    self.addCleanup(revtree.unlock)
    old_inv = tree.branch.repository.revision_tree(revision.NULL_REVISION).root_inventory
    new_inv = revtree.root_inventory
    delta = self.make_inv_delta(old_inv, new_inv)
    repo_direct = self._get_repo_in_write_group('direct')
    add_validator = repo_direct.add_inventory(revid, new_inv, [])
    repo_direct.commit_write_group()
    repo_delta = self._get_repo_in_write_group('delta')
    try:
        delta_validator, inv = repo_delta.add_inventory_by_delta(revision.NULL_REVISION, delta, revid, [])
    except:
        repo_delta.abort_write_group()
        raise
    else:
        repo_delta.commit_write_group()
    self.assertEqual(add_validator, delta_validator)
    self.assertEqual(list(new_inv.iter_entries()), list(inv.iter_entries()))