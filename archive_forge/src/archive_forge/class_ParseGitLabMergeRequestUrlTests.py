from datetime import datetime
from breezy.plugins.gitlab.forge import (NotGitLabUrl, NotMergeRequestUrl,
from breezy.tests import TestCase
class ParseGitLabMergeRequestUrlTests(TestCase):

    def test_invalid(self):
        self.assertRaises(NotMergeRequestUrl, parse_gitlab_merge_request_url, 'https://salsa.debian.org/')
        self.assertRaises(NotGitLabUrl, parse_gitlab_merge_request_url, 'bzr://salsa.debian.org/')
        self.assertRaises(NotGitLabUrl, parse_gitlab_merge_request_url, 'https:///salsa.debian.org/')
        self.assertRaises(NotMergeRequestUrl, parse_gitlab_merge_request_url, 'https://salsa.debian.org/jelmer/salsa')

    def test_old_style(self):
        self.assertEqual(('salsa.debian.org', 'jelmer/salsa', 4), parse_gitlab_merge_request_url('https://salsa.debian.org/jelmer/salsa/merge_requests/4'))

    def test_new_style(self):
        self.assertEqual(('salsa.debian.org', 'jelmer/salsa', 4), parse_gitlab_merge_request_url('https://salsa.debian.org/jelmer/salsa/-/merge_requests/4'))