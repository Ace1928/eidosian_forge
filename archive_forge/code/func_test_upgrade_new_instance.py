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
def test_upgrade_new_instance(self):
    """Does an available updater work?"""
    dir = self.make_controldir('.')
    dir.create_repository()
    dir.create_branch()
    self.createWorkingTreeOrSkip(dir)
    if dir.can_convert_format():
        with ui.ui_factory.nested_progress_bar() as pb:
            dir._format.get_converter(format=dir._format).convert(dir, pb)
        check.check_dwim(self.get_url('.'), False, True, True)