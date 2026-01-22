from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class IncorrectlyOrderedParentsScenario(BrokenRepoScenario):
    """A scenario where the set parents of a version of a file are correct, but
    the order of those parents is incorrect.

    This defines a 'broken-revision-1-2' and a 'broken-revision-2-1' which both
    have their file version parents reversed compared to the revision parents,
    which is invalid.  (We use two revisions with opposite orderings of the
    same parents to make sure that accidentally relying on dictionary/set
    ordering cannot make the test pass; the assumption is that while dict/set
    iteration order is arbitrary, it is also consistent within a single test).
    """

    def all_versions_after_reconcile(self):
        return [b'parent-1', b'parent-2', b'broken-revision-1-2', b'broken-revision-2-1']

    def populated_parents(self):
        return (((), b'parent-1'), ((), b'parent-2'), ((b'parent-2', b'parent-1'), b'broken-revision-1-2'), ((b'parent-1', b'parent-2'), b'broken-revision-2-1'))

    def corrected_parents(self):
        return (((), b'parent-1'), ((), b'parent-2'), ((b'parent-1', b'parent-2'), b'broken-revision-1-2'), ((b'parent-2', b'parent-1'), b'broken-revision-2-1'))

    def check_regexes(self, repo):
        if repo.supports_rich_root():
            count = 4
        else:
            count = 2
        return ('%d inconsistent parents' % count, '\\* a-file-id version broken-revision-1-2 has parents \\(parent-2, parent-1\\) but should have \\(parent-1, parent-2\\)', '\\* a-file-id version broken-revision-2-1 has parents \\(parent-1, parent-2\\) but should have \\(parent-2, parent-1\\)')

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'parent-1', [])
        self.add_revision(repo, b'parent-1', inv, [])
        inv = self.make_one_file_inventory(repo, b'parent-2', [])
        self.add_revision(repo, b'parent-2', inv, [])
        inv = self.make_one_file_inventory(repo, b'broken-revision-1-2', [b'parent-2', b'parent-1'])
        self.add_revision(repo, b'broken-revision-1-2', inv, [b'parent-1', b'parent-2'])
        inv = self.make_one_file_inventory(repo, b'broken-revision-2-1', [b'parent-1', b'parent-2'])
        self.add_revision(repo, b'broken-revision-2-1', inv, [b'parent-2', b'parent-1'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'broken-revision-1-2'): True, (b'TREE_ROOT', b'broken-revision-2-1'): True, (b'TREE_ROOT', b'parent-1'): True, (b'TREE_ROOT', b'parent-2'): True})
        result.update({(b'a-file-id', b'broken-revision-1-2'): True, (b'a-file-id', b'broken-revision-2-1'): True, (b'a-file-id', b'parent-1'): True, (b'a-file-id', b'parent-2'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'broken-revision-1-2'): [(b'a-file-id', b'parent-1'), (b'a-file-id', b'parent-2')], (b'a-file-id', b'broken-revision-2-1'): [(b'a-file-id', b'parent-2'), (b'a-file-id', b'parent-1')], (b'a-file-id', b'parent-1'): [NULL_REVISION], (b'a-file-id', b'parent-2'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'broken-revision-1-2'): [(b'TREE_ROOT', b'parent-1'), (b'TREE_ROOT', b'parent-2')], (b'TREE_ROOT', b'broken-revision-2-1'): [(b'TREE_ROOT', b'parent-2'), (b'TREE_ROOT', b'parent-1')], (b'TREE_ROOT', b'parent-1'): [NULL_REVISION], (b'TREE_ROOT', b'parent-2'): [NULL_REVISION]}