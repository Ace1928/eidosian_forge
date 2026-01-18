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
def test_strict_history(self):
    tree1 = self.make_branch_and_tree('tree1')
    try:
        tree1.branch.set_append_revisions_only(True)
    except errors.UpgradeRequired:
        raise tests.TestSkipped('Format does not support strict history')
    tree1.commit('empty commit')
    tree2 = tree1.controldir.sprout('tree2').open_workingtree()
    tree2.commit('empty commit 2')
    tree1.pull(tree2.branch)
    tree1.commit('empty commit 3')
    tree2.commit('empty commit 4')
    self.assertRaises(errors.DivergedBranches, tree1.pull, tree2.branch)
    tree2.merge_from_branch(tree1.branch)
    tree2.commit('empty commit 5')
    self.assertRaises(errors.AppendRevisionsOnlyViolation, tree1.pull, tree2.branch)
    tree3 = tree1.controldir.sprout('tree3').open_workingtree()
    tree3.merge_from_branch(tree2.branch)
    tree3.commit('empty commit 6')
    tree2.pull(tree3.branch)