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
def test_push_within_repository(self):
    """Push from one branch to another inside the same repository."""
    try:
        repo = self.make_repository('repo', shared=True)
    except (errors.IncompatibleFormat, errors.UninitializableFormat):
        return
    try:
        a_branch = self.make_from_branch('repo/tree')
    except errors.UninitializableFormat:
        return
    try:
        tree = a_branch.controldir.create_workingtree()
    except errors.UnsupportedOperation:
        self.assertFalse(a_branch.controldir._format.supports_workingtrees)
        tree = a_branch.create_checkout('repo/tree', lightweight=True)
    except errors.NotLocalUrl:
        if self.vfs_transport_factory is test_server.LocalURLServer:
            local_controldir = controldir.ControlDir.open(self.get_vfs_only_url('repo/tree'))
            tree = local_controldir.create_workingtree()
        else:
            tree = a_branch.create_checkout('repo/tree', lightweight=True)
    self.build_tree(['repo/tree/a'])
    tree.add(['a'])
    tree.commit('a')
    to_branch = self.make_to_branch('repo/branch')
    try:
        tree.branch.push(to_branch)
    except errors.NoRoundtrippingSupport:
        tree.branch.push(to_branch, lossy=True)
    else:
        self.assertEqual(tree.branch.last_revision(), to_branch.last_revision())