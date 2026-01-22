from breezy import errors, revision
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def _get_repo_in_write_group(self, path='repository'):
    repo = self.make_repository(path)
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    return repo