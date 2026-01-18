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
def test_search_missing_revision_ids_limit(self):
    repo_b = self.make_to_repository('rev1_only')
    repo_a = self.controldir.open_repository()
    self.assertFalse(repo_b.has_revision(self.rev2))
    try:
        result = repo_b.search_missing_revision_ids(repo_a, limit=1)
    except errors.FetchLimitUnsupported:
        raise TestNotApplicable('interrepo does not support limited fetches')
    self.assertEqual(('search', {self.rev1}, {b'null:'}, 1), result.get_recipe())