from breezy import branchbuilder, errors, tests, urlutils
from breezy.branch import Branch
from breezy.controldir import BranchReferenceLoop
from breezy.tests import per_controldir
from breezy.tests.features import UnicodeFilenameFeature
def test_sprout_into_colocated(self):
    if not self.bzrdir_format.is_supported():
        raise tests.TestNotApplicable('Control dir format not supported')
    from_tree = self.make_branch_and_tree('from')
    revid = from_tree.commit('rev1')
    try:
        other_branch = self.make_branch('to')
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('Control dir does not support creating new branches.')
    to_dir = from_tree.controldir.sprout(urlutils.join_segment_parameters(other_branch.user_url, {'branch': 'target'}))
    to_branch = to_dir.open_branch(name='target')
    self.assertEqual(revid, to_branch.last_revision())