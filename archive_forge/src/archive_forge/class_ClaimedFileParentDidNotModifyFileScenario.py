from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class ClaimedFileParentDidNotModifyFileScenario(BrokenRepoScenario):
    """A scenario where the file parent is the same as the revision parent, but
    should not be because that revision did not modify the file.

    Specifically, the parent revision of 'current' is
    'modified-something-else', which does not modify 'a-file', but the
    'current' version of 'a-file' erroneously claims that
    'modified-something-else' is the parent file version.
    """

    def all_versions_after_reconcile(self):
        return (b'basis', b'current')

    def populated_parents(self):
        return (((), b'basis'), ((b'basis',), b'modified-something-else'), ((b'modified-something-else',), b'current'))

    def corrected_parents(self):
        return (((), b'basis'), (None, b'modified-something-else'), ((b'basis',), b'current'))

    def check_regexes(self, repo):
        if repo.supports_rich_root():
            count = 3
        else:
            count = 1
        return ('%d inconsistent parents' % count, '\\* a-file-id version current has parents \\(modified-something-else\\) but should have \\(basis\\)')

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'basis', ())
        self.add_revision(repo, b'basis', inv, ())
        inv = self.make_one_file_inventory(repo, b'modified-something-else', (b'basis',), inv_revision=b'basis')
        self.add_revision(repo, b'modified-something-else', inv, (b'basis',))
        inv = self.make_one_file_inventory(repo, b'current', (b'modified-something-else',))
        self.add_revision(repo, b'current', inv, (b'modified-something-else',))
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'basis'): True, (b'TREE_ROOT', b'current'): True, (b'TREE_ROOT', b'modified-something-else'): True})
        result.update({(b'a-file-id', b'basis'): True, (b'a-file-id', b'current'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'basis'): [NULL_REVISION], (b'a-file-id', b'current'): [(b'a-file-id', b'basis')]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'basis'): [b'null:'], (b'TREE_ROOT', b'current'): [(b'TREE_ROOT', b'modified-something-else')], (b'TREE_ROOT', b'modified-something-else'): [(b'TREE_ROOT', b'basis')]}