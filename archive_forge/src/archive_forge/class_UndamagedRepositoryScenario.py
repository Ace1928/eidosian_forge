from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class UndamagedRepositoryScenario(BrokenRepoScenario):
    """A scenario where the repository has no damage.

    It has a single revision, 'rev1a', with a single file.
    """

    def all_versions_after_reconcile(self):
        return (b'rev1a',)

    def populated_parents(self):
        return (((), b'rev1a'),)

    def corrected_parents(self):
        return self.populated_parents()

    def check_regexes(self, repo):
        return ['0 unreferenced text versions']

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'rev1a', [], root_revision=b'rev1a')
        self.add_revision(repo, b'rev1a', inv, [])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'rev1a'): True})
        result.update({(b'a-file-id', b'rev1a'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'rev1a'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'rev1a'): [NULL_REVISION]}