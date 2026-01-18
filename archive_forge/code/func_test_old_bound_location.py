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
def test_old_bound_location(self):
    branch = self.make_branch('branch1')
    try:
        self.assertIs(None, branch.get_old_bound_location())
    except errors.UpgradeRequired:
        raise tests.TestNotApplicable('Format does not store old bound locations')
    branch2 = self.make_branch('branch2')
    branch.bind(branch2)
    self.assertIs(None, branch.get_old_bound_location())
    branch.unbind()
    self.assertContainsRe(branch.get_old_bound_location(), '\\/branch2\\/$')