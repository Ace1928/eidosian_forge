from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class TooManyParentsScenario(BrokenRepoScenario):
    """A scenario where 'broken-revision' of 'a-file' claims to have parents
    ['good-parent', 'bad-parent'].  However 'bad-parent' is in the ancestry of
    'good-parent', so the correct parent list for that file version are is just
    ['good-parent'].
    """

    def all_versions_after_reconcile(self):
        return (b'bad-parent', b'good-parent', b'broken-revision')

    def populated_parents(self):
        return (((), b'bad-parent'), ((b'bad-parent',), b'good-parent'), ((b'good-parent', b'bad-parent'), b'broken-revision'))

    def corrected_parents(self):
        return (((), b'bad-parent'), ((b'bad-parent',), b'good-parent'), ((b'good-parent',), b'broken-revision'))

    def check_regexes(self, repo):
        if repo.supports_rich_root():
            count = 3
        else:
            count = 1
        return ('     %d inconsistent parents' % count, '      \\* a-file-id version broken-revision has parents \\(good-parent, bad-parent\\) but should have \\(good-parent\\)')

    def populate_repository(self, repo):
        inv = self.make_one_file_inventory(repo, b'bad-parent', (), root_revision=b'bad-parent')
        self.add_revision(repo, b'bad-parent', inv, ())
        inv = self.make_one_file_inventory(repo, b'good-parent', (b'bad-parent',))
        self.add_revision(repo, b'good-parent', inv, (b'bad-parent',))
        inv = self.make_one_file_inventory(repo, b'broken-revision', (b'good-parent', b'bad-parent'))
        self.add_revision(repo, b'broken-revision', inv, (b'good-parent',))
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        result = {}
        if self.versioned_root:
            result.update({(b'TREE_ROOT', b'bad-parent'): True, (b'TREE_ROOT', b'broken-revision'): True, (b'TREE_ROOT', b'good-parent'): True})
        result.update({(b'a-file-id', b'bad-parent'): True, (b'a-file-id', b'broken-revision'): True, (b'a-file-id', b'good-parent'): True})
        return result

    def repository_text_keys(self):
        return {(b'a-file-id', b'bad-parent'): [NULL_REVISION], (b'a-file-id', b'broken-revision'): [(b'a-file-id', b'good-parent')], (b'a-file-id', b'good-parent'): [(b'a-file-id', b'bad-parent')]}

    def versioned_repository_text_keys(self):
        return {(b'TREE_ROOT', b'bad-parent'): [NULL_REVISION], (b'TREE_ROOT', b'broken-revision'): [(b'TREE_ROOT', b'good-parent')], (b'TREE_ROOT', b'good-parent'): [(b'TREE_ROOT', b'bad-parent')]}