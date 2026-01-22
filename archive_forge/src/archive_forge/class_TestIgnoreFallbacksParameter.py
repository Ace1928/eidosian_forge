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
class TestIgnoreFallbacksParameter(per_branch.TestCaseWithBranch):

    def make_branch_with_fallback(self):
        fallback = self.make_branch('fallback')
        if not fallback._format.supports_stacking():
            raise tests.TestNotApplicable('format does not support stacking')
        stacked = self.make_branch('stacked')
        stacked.set_stacked_on_url(fallback.base)
        return stacked

    def test_fallbacks_not_opened(self):
        stacked = self.make_branch_with_fallback()
        self.get_transport('').rename('fallback', 'moved')
        reopened_dir = controldir.ControlDir.open(stacked.base)
        reopened = reopened_dir.open_branch(ignore_fallbacks=True)
        self.assertEqual([], reopened.repository._fallback_repositories)

    def test_fallbacks_are_opened(self):
        stacked = self.make_branch_with_fallback()
        reopened_dir = controldir.ControlDir.open(stacked.base)
        reopened = reopened_dir.open_branch(ignore_fallbacks=False)
        self.assertLength(1, reopened.repository._fallback_repositories)