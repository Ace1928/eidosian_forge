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
def test_open_repository(self):
    if not self.bzrdir_format.is_initializable():
        raise TestNotApplicable('format is not initializable')
    t = self.get_transport()
    made_control = self.bzrdir_format.initialize(t.base)
    made_repo = made_control.create_repository()
    opened_repo = made_control.open_repository()
    self.assertEqual(made_control, opened_repo.controldir)
    self.assertIsInstance(opened_repo, made_repo.__class__)
    self.assertIsInstance(opened_repo._format, made_repo._format.__class__)