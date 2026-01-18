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
def test_set_branch_reference_on_existing_branch(self):
    """set_branch_reference creates a branch reference"""
    referenced_branch = self.make_branch('referenced')
    dir = self.make_branch('source').controldir
    try:
        reference = dir.set_branch_reference(referenced_branch)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('control directory does not support branch references')
    self.assertEqual(referenced_branch.user_url, dir.get_branch_reference())