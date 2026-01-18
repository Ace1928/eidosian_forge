import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_sprout_branch_no_source_branch(self):
    try:
        repo = self.make_repository('source', shared=True)
    except errors.IncompatibleFormat:
        return
    if isinstance(self.bzrdir_format, RemoteBzrDirFormat):
        self.skipTest('remote formats not supported')
    branch = controldir.ControlDir.create_branch_convenience('source/trunk')
    tree = branch.controldir.open_workingtree()
    self.build_tree(['source/trunk/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    rev2 = tree.commit('revision 2', allow_pointless=True)
    target = self.sproutOrSkip(repo.controldir, self.get_url('target'), revision_id=rev2)
    self.assertEqual([rev2], target.open_workingtree().get_parent_ids())