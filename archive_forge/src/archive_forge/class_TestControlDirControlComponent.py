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
class TestControlDirControlComponent(TestCaseWithControlDir):
    """ControlDir implementations adequately implement ControlComponent."""

    def test_urls(self):
        bd = self.make_controldir('bd')
        self.assertIsInstance(bd.user_url, str)
        self.assertEqual(bd.user_url, bd.user_transport.base)
        self.assertEqual(bd.control_url.find(bd.user_url), 0)
        self.assertEqual(bd.control_url, bd.control_transport.base)