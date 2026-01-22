import sys
import breezy
import breezy.errors as errors
import breezy.gpg
from breezy.bzr.inventory import Inventory
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.workingtree import WorkingTree
class TestCaseWithGhosts(TestCaseWithInterRepository):

    def test_fetch_all_fixes_up_ghost(self):
        has_ghost = self.make_repository('has_ghost')
        missing_ghost = self.make_repository('missing_ghost')
        if [True, True] != [repo._format.supports_ghosts for repo in (has_ghost, missing_ghost)]:
            raise TestNotApplicable('Need ghost support.')

        def add_commit(repo, revision_id, parent_ids):
            repo.lock_write()
            repo.start_write_group()
            inv = Inventory(revision_id=revision_id)
            inv.root.revision = revision_id
            root_id = inv.root.file_id
            sha1 = repo.add_inventory(revision_id, inv, parent_ids)
            repo.texts.add_lines((root_id, revision_id), [], [])
            rev = breezy.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
            rev.parent_ids = parent_ids
            repo.add_revision(revision_id, rev)
            repo.commit_write_group()
            repo.unlock()
        add_commit(has_ghost, b'ghost', [])
        add_commit(has_ghost, b'references', [b'ghost'])
        add_commit(missing_ghost, b'references', [b'ghost'])
        add_commit(has_ghost, b'tip', [b'references'])
        missing_ghost.fetch(has_ghost, b'tip', find_ghosts=True)
        rev = missing_ghost.get_revision(b'tip')
        inv = missing_ghost.get_inventory(b'tip')
        rev = missing_ghost.get_revision(b'ghost')
        inv = missing_ghost.get_inventory(b'ghost')
        self.assertThat([b'ghost', b'references', b'tip'], MatchesAncestry(missing_ghost, b'tip'))