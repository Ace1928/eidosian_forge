from io import BytesIO
from testtools.matchers import Equals, MatchesAny
from ... import branch, check, controldir, errors, push, tests
from ...branch import BindingUnsupported, Branch
from ...bzr import branch as bzrbranch
from ...bzr import vf_repository
from ...bzr.smart.repository import SmartServerRepositoryGetParentMap
from ...controldir import ControlDir
from ...revision import NULL_REVISION
from .. import test_server
from . import TestCaseWithInterBranch
def test_push_raises_specific_error_on_master_connection_error(self):
    master_tree = self.make_to_branch_and_tree('master')
    checkout = self.make_to_branch_and_tree('checkout')
    try:
        checkout.branch.bind(master_tree.branch)
    except BindingUnsupported:
        return
    other_bzrdir = self.sprout_from(master_tree.branch.controldir, 'other')
    other = other_bzrdir.open_workingtree()
    master_tree.controldir.destroy_branch()
    self.assertRaises(errors.BoundBranchConnectionFailure, other.branch.push, checkout.branch)