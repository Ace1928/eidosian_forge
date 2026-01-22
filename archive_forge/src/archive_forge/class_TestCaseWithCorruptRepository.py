from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestCaseWithCorruptRepository(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        super().setUp()
        repo = self.make_repository('inventory_with_unnecessary_ghost')
        repo.lock_write()
        repo.start_write_group()
        inv = inventory.Inventory(revision_id=b'ghost')
        inv.root.revision = b'ghost'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, b'ghost'), [], [])
        sha1 = repo.add_inventory(b'ghost', inv, [])
        rev = _mod_revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=b'ghost')
        rev.parent_ids = [b'the_ghost']
        try:
            repo.add_revision(b'ghost', rev)
        except (errors.NoSuchRevision, errors.RevisionNotPresent):
            raise tests.TestNotApplicable('Cannot test with ghosts for this format.')
        inv = inventory.Inventory(revision_id=b'the_ghost')
        inv.root.revision = b'the_ghost'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, b'the_ghost'), [], [])
        sha1 = repo.add_inventory(b'the_ghost', inv, [])
        rev = _mod_revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=b'the_ghost')
        rev.parent_ids = []
        repo.add_revision(b'the_ghost', rev)
        inv_weave = repo.inventories
        possible_parents = (None, ((b'ghost',),))
        self.assertSubset(inv_weave.get_parent_map([(b'ghost',)])[b'ghost',], possible_parents)
        repo.commit_write_group()
        repo.unlock()

    def test_corrupt_revision_access_asserts_if_reported_wrong(self):
        repo_url = self.get_url('inventory_with_unnecessary_ghost')
        repo = _mod_repository.Repository.open(repo_url)
        m = MatchesAncestry(repo, b'ghost')
        reported_wrong = False
        try:
            if m.match([b'the_ghost', b'ghost']) is not None:
                reported_wrong = True
        except errors.CorruptRepository:
            return
        if not reported_wrong:
            return
        self.assertRaises(errors.CorruptRepository, repo.get_revision, b'ghost')

    def test_corrupt_revision_get_revision_reconcile(self):
        repo_url = self.get_url('inventory_with_unnecessary_ghost')
        repo = _mod_repository.Repository.open(repo_url)
        repo.get_revision_reconcile(b'ghost')