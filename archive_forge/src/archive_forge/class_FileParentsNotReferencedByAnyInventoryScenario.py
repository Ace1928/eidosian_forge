from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class FileParentsNotReferencedByAnyInventoryScenario(BrokenRepoScenario):
    """A scenario where a repository with file 'a-file' which has extra
    per-file versions that are not referenced by any inventory (even though
    they have the same ID as actual revisions).  The inventory of 'rev2'
    references 'rev1a' of 'a-file', but there is a 'rev2' of 'some-file' stored
    and erroneously referenced by later per-file versions (revisions 'rev4' and
    'rev5').

    Reconcile should remove the file parents that are not referenced by any
    inventory.
    """

    def all_versions_after_reconcile(self):
        return (b'rev1a', b'rev2c', b'rev4', b'rev5')

    def populated_parents(self):
        return [((b'rev1a',), b'rev2'), ((b'rev1a',), b'rev2b'), ((b'rev2',), b'rev3'), ((b'rev2',), b'rev4'), ((b'rev2', b'rev2c'), b'rev5')]

    def corrected_parents(self):
        return ((None, b'rev2'), (None, b'rev2b'), ((b'rev1a',), b'rev3'), ((b'rev1a',), b'rev4'), ((b'rev2c',), b'rev5'))

    def check_regexes(self, repo):
        if repo.supports_rich_root():
            count = 9
        else:
            count = 3
        return ['unreferenced version: {rev2} in a-file-id', 'unreferenced version: {rev2b} in a-file-id', 'a-file-id version rev3 has parents \\(rev2\\) but should have \\(rev1a\\)', 'a-file-id version rev5 has parents \\(rev2, rev2c\\) but should have \\(rev2c\\)', 'a-file-id version rev4 has parents \\(rev2\\) but should have \\(rev1a\\)', '%d inconsistent parents' % count]

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'rev1a', [], root_revision=b'rev1a')
        self.add_revision(repo, b'rev1a', inv, [])
        self.make_one_file_inventory(repo, b'rev2', [b'rev1a'], inv_revision=b'rev1a')
        self.add_revision(repo, b'rev2', inv, [b'rev1a'])
        inv = self.make_one_file_inventory(repo, b'rev3', [b'rev2'])
        self.add_revision(repo, b'rev3', inv, [b'rev1c', b'rev1a'])
        inv = self.make_one_file_inventory(repo, b'rev2b', [b'rev1a'], inv_revision=b'rev1a')
        self.add_revision(repo, b'rev2b', inv, [b'rev1a'])
        inv = self.make_one_file_inventory(repo, b'rev4', [b'rev2'])
        self.add_revision(repo, b'rev4', inv, [b'rev2', b'rev2b'])
        inv = self.make_one_file_inventory(repo, b'rev2c', [b'rev1a'])
        self.add_revision(repo, b'rev2c', inv, [b'rev1a'])
        inv = self.make_one_file_inventory(repo, b'rev5', [b'rev2', b'rev2c'])
        self.add_revision(repo, b'rev5', inv, [b'rev2', b'rev2c'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'rev1a'): True, (b'TREE_ROOT', b'rev2'): True, (b'TREE_ROOT', b'rev2b'): True, (b'TREE_ROOT', b'rev2c'): True, (b'TREE_ROOT', b'rev3'): True, (b'TREE_ROOT', b'rev4'): True, (b'TREE_ROOT', b'rev5'): True})
        result.update({(b'a-file-id', b'rev1a'): True, (b'a-file-id', b'rev2c'): True, (b'a-file-id', b'rev3'): True, (b'a-file-id', b'rev4'): True, (b'a-file-id', b'rev5'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'rev1a'): [NULL_REVISION], (b'a-file-id', b'rev2c'): [(b'a-file-id', b'rev1a')], (b'a-file-id', b'rev3'): [(b'a-file-id', b'rev1a')], (b'a-file-id', b'rev4'): [(b'a-file-id', b'rev1a')], (b'a-file-id', b'rev5'): [(b'a-file-id', b'rev2c')]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'rev1a'): [NULL_REVISION], (b'TREE_ROOT', b'rev2'): [(b'TREE_ROOT', b'rev1a')], (b'TREE_ROOT', b'rev2b'): [(b'TREE_ROOT', b'rev1a')], (b'TREE_ROOT', b'rev2c'): [(b'TREE_ROOT', b'rev1a')], (b'TREE_ROOT', b'rev3'): [(b'TREE_ROOT', b'rev1a')], (b'TREE_ROOT', b'rev4'): [(b'TREE_ROOT', b'rev2'), (b'TREE_ROOT', b'rev2b')], (b'TREE_ROOT', b'rev5'): [(b'TREE_ROOT', b'rev2'), (b'TREE_ROOT', b'rev2c')]}