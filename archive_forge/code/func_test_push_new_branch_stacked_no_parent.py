import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_new_branch_stacked_no_parent(self):
    """Pushing with --stacked and no parent branch errors."""
    branch = self.make_branch_and_tree('branch', format='1.9')
    out, err = self.run_bzr_error(['Could not determine branch to refer to\\.'], ['push', '--stacked', self.get_url('published')], working_dir='branch')
    self.assertEqual('', out)
    self.assertFalse(self.get_transport('published').has('.'))