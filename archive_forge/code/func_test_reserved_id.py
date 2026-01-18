from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_reserved_id(self):
    repo = self.make_repository('repository')
    with repo.lock_write(), _mod_repository.WriteGroup(repo):
        self.assertRaises(errors.ReservedId, repo.add_inventory, b'reserved:', None, None)
        self.assertRaises(errors.ReservedId, repo.add_inventory_by_delta, 'foo', [], b'reserved:', None)
        self.assertRaises(errors.ReservedId, repo.add_revision, b'reserved:', None)