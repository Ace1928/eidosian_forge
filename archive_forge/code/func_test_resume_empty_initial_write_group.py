import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_resume_empty_initial_write_group(self):
    """Resuming an empty token list is equivalent to start_write_group."""
    self.require_suspendable_write_groups('Cannot test resume on repo that does not support suspending')
    repo = self.make_write_locked_repo()
    repo.resume_write_group([])
    repo.abort_write_group()