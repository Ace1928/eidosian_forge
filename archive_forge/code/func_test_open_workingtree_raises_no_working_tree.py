import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_open_workingtree_raises_no_working_tree(self):
    """ControlDir.open_workingtree() should raise NoWorkingTree (rather than
        e.g. NotLocalUrl) if there is no working tree.
        """
    dir = self.make_controldir('source')
    vfs_dir = controldir.ControlDir.open(self.get_vfs_only_url('source'))
    if vfs_dir.has_workingtree():
        raise TestNotApplicable('format does not support control directories without working tree')
    self.assertRaises(errors.NoWorkingTree, dir.open_workingtree)