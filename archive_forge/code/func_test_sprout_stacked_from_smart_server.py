from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_stacked_from_smart_server(self):
    trunk_tree = self.make_branch_and_tree('mainline')
    trunk_revid = trunk_tree.commit('mainline')
    try:
        trunk_tree.controldir.sprout('testbranch', stacked=True)
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    remote_transport = self.make_smart_server('mainline')
    remote_bzrdir = controldir.ControlDir.open_from_transport(remote_transport)
    new_dir = remote_bzrdir.sprout('newbranch', stacked=True)
    self.assertRevisionNotInRepository('newbranch', trunk_revid)
    tree = new_dir.open_branch().create_checkout('local')
    new_branch_revid = tree.commit('something local')
    self.assertRevisionNotInRepository(trunk_tree.branch.user_url, new_branch_revid)
    self.assertRevisionInRepository('newbranch', new_branch_revid)