import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
class TestBranchControlComponent(per_branch.TestCaseWithBranch):
    """Branch implementations adequately implement ControlComponent."""

    def test_urls(self):
        br = self.make_branch('branch')
        self.assertIsInstance(br.user_url, str)
        self.assertEqual(br.user_url, br.user_transport.base)
        self.assertEqual(br.control_url, br.control_transport.base)