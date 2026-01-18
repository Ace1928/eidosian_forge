from breezy import errors, repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_refresh_data_after_fetch_new_data_visible_in_write_group(self):
    tree = self.make_branch_and_memory_tree('target')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add([''], ids=[b'root-id'])
    tree.commit('foo', rev_id=b'commit-in-target')
    repo = tree.branch.repository
    token = repo.lock_write().repository_token
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    self.addCleanup(repo.abort_write_group)
    self.fetch_new_revision_into_concurrent_instance(repo, token)
    try:
        repo.refresh_data()
    except repository.IsInWriteGroupError:
        pass
    else:
        self.assertEqual([b'commit-in-target', b'new-rev'], sorted(repo.all_revision_ids()))