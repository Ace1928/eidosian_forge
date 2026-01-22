import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def assertPublished(self, branch_revid, stacked_on):
    """Assert that the branch 'published' has been published correctly."""
    published_branch = branch.Branch.open('published')
    self.assertEqual(stacked_on, published_branch.get_stacked_on_url())
    self.assertTrue(published_branch.repository.has_revision(branch_revid))