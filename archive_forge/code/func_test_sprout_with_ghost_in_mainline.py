import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_with_ghost_in_mainline(self):
    tree = self.make_branch_and_tree('tree1')
    if not tree.branch.repository._format.supports_ghosts:
        raise tests.TestNotApplicable('repository format does not support ghosts in mainline')
    tree.set_parent_ids([b'spooky'], allow_leftmost_as_ghost=True)
    tree.add('')
    rev1 = tree.commit('msg1')
    tree.commit('msg2')
    tree.controldir.sprout('target', revision_id=rev1)