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
def test_search_missing_revision_ids(self):
    repo_b = self.make_to_repository('rev1_only')
    repo_a = self.controldir.open_repository()
    try:
        repo_b.fetch(repo_a, self.rev1)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    self.assertFalse(repo_b.has_revision(self.rev2))
    result = repo_b.search_missing_revision_ids(repo_a)
    self.assertEqual({self.rev2}, result.get_keys())
    self.assertEqual(('search', {self.rev2}, {self.rev1}, 1), result.get_recipe())