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
def test_sprout_with_revision_id_uses_default_stack_on(self):
    builder = self.make_branch_builder('stack-on')
    builder.start_series()
    rev1 = builder.build_commit(message='Rev 1.')
    rev2 = builder.build_commit(message='Rev 2.')
    rev3 = builder.build_commit(message='Rev 3.')
    builder.finish_series()
    stack_on = builder.get_branch()
    config = self.make_controldir('policy-dir').get_config()
    try:
        config.set_default_stack_on(self.get_url('stack-on'))
    except errors.BzrError:
        raise TestNotApplicable('Only relevant for stackable formats.')
    sprouted = stack_on.controldir.sprout(self.get_url('policy-dir/sprouted'), revision_id=rev3)
    repo = sprouted.open_repository()
    self.addCleanup(repo.lock_read().unlock)
    self.assertEqual(None, repo.get_parent_map([rev1]).get(rev1))