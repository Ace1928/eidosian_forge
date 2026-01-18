from datetime import datetime
from breezy.plugins.gitlab.forge import (NotGitLabUrl, NotMergeRequestUrl,
from breezy.tests import TestCase
def test_old_style(self):
    self.assertEqual(('salsa.debian.org', 'jelmer/salsa', 4), parse_gitlab_merge_request_url('https://salsa.debian.org/jelmer/salsa/merge_requests/4'))