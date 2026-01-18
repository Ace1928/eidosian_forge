from breezy import errors, repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_refresh_data_after_fetch_new_data_visible(self):
    repo = self.make_repository('target')
    token = repo.lock_write().repository_token
    self.addCleanup(repo.unlock)
    self.fetch_new_revision_into_concurrent_instance(repo, token)
    repo.refresh_data()
    self.assertNotEqual({}, repo.get_graph().get_parent_map([b'new-rev']))