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
def test_sprout_controldir_tree_branch_reference(self):
    referenced_branch = self.make_branch('referencced')
    dir = self.make_controldir('source')
    try:
        dir.set_branch_reference(referenced_branch)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('format does not support branch references')
    self.assertRaises(errors.NoRepositoryPresent, dir.open_repository)
    tree = self.createWorkingTreeOrSkip(dir)
    self.build_tree(['source/subdir/'])
    tree.add('subdir')
    target = dir.sprout(self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertEqual(target, target.open_branch().controldir)
    target.open_repository()
    result_tree = target.open_workingtree()
    self.assertFalse(result_tree.has_filename('subdir'))