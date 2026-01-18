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
def test_store_uncommitted_bound(self):
    tree = self.make_branch_and_tree('b')
    branch = tree.branch
    master = self.make_branch('master')
    self.bind(branch, master)
    creator = FakeShelfCreator(tree.branch)
    self.assertIs(None, tree.branch.get_unshelver(tree))
    self.assertIs(None, master.get_unshelver(tree))
    tree.branch.store_uncommitted(creator)
    self.assertIsNot(None, master.get_unshelver(tree))