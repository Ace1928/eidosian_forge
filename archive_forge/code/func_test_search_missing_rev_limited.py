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
def test_search_missing_rev_limited(self):
    repo_b = self.make_to_repository('empty')
    repo_a = self.controldir.open_repository()
    result = repo_b.search_missing_revision_ids(repo_a, revision_ids=[self.rev1])
    self.assertEqual({self.rev1}, result.get_keys())
    self.assertEqual(('search', {self.rev1}, {NULL_REVISION}, 1), result.get_recipe())