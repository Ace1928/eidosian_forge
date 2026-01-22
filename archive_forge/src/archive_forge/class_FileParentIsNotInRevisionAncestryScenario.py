from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class FileParentIsNotInRevisionAncestryScenario(BrokenRepoScenario):
    """A scenario where a revision 'rev2' has 'a-file' with a
    parent 'rev1b' that is not in the revision ancestry.

    Reconcile should remove 'rev1b' from the parents list of 'a-file' in
    'rev2', preserving 'rev1a' as a parent.
    """

    def all_versions_after_reconcile(self):
        return (b'rev1a', b'rev2')

    def populated_parents(self):
        return (((), b'rev1a'), ((), b'rev1b'), ((b'rev1a', b'rev1b'), b'rev2'))

    def corrected_parents(self):
        return (((), b'rev1a'), (None, b'rev1b'), ((b'rev1a',), b'rev2'))

    def check_regexes(self, repo):
        return ['\\* a-file-id version rev2 has parents \\(rev1a, rev1b\\) but should have \\(rev1a\\)', '1 unreferenced text versions']

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'rev1a', [], root_revision=b'rev1a')
        self.add_revision(repo, b'rev1a', inv, [])
        inv = self.make_one_file_inventory(repo, b'rev1b', [], root_revision=b'rev1b')
        repo.add_inventory(b'rev1b', inv, [])
        inv = self.make_one_file_inventory(repo, b'rev2', [b'rev1a', b'rev1b'])
        self.add_revision(repo, b'rev2', inv, [b'rev1a'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'rev1a'): True, (b'TREE_ROOT', b'rev2'): True})
        result.update({(b'a-file-id', b'rev1a'): True, (b'a-file-id', b'rev2'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'rev1a'): [NULL_REVISION], (b'a-file-id', b'rev2'): [(b'a-file-id', b'rev1a')]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'rev1a'): [NULL_REVISION], (b'TREE_ROOT', b'rev2'): [(b'TREE_ROOT', b'rev1a')]}