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
def test_store_uncommitted_none(self):
    branch = self.make_branch('b')
    with skip_if_storing_uncommitted_unsupported():
        branch.store_uncommitted(FakeShelfCreator(branch))
    branch.store_uncommitted(None)
    self.assertIs(None, branch.get_unshelver(None))