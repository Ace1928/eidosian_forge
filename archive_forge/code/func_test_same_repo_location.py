from breezy import controldir, transport
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_same_repo_location(self):
    """Different repository objects for the same location are the same."""
    repo = self.make_repository('.')
    reopened_repo = repo.controldir.open_repository()
    self.assertFalse(repo is reopened_repo, 'This test depends on reopened_repo being a different instance of the same repo.')
    self.assertSameRepo(repo, reopened_repo)