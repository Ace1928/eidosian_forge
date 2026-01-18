import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_partial(self):
    wt_a = self.make_branch_and_tree('a')
    self.build_tree(['a/one'])
    wt_a.add(['one'])
    rev1 = wt_a.commit('commit one')
    self.build_tree(['a/two'])
    wt_a.add(['two'])
    wt_a.commit('commit two')
    repo_b = self.make_repository('b')
    repo_a = wt_a.branch.repository
    repo_a.copy_content_into(repo_b)
    br_b = wt_a.branch.sprout(repo_b.controldir, revision_id=rev1)
    self.assertEqual(rev1, br_b.last_revision())