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
def test_sprout_controldir_repository_branch_only_source_under_shared(self):
    try:
        shared_repo = self.make_repository('shared', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('format does not support shared repositories')
    if not shared_repo._format.supports_nesting_repositories:
        raise TestNotApplicable('format does not support nesting repositories')
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    rev1 = tree.commit('revision 1')
    tree.controldir.open_branch().generate_revision_history(_mod_revision.NULL_REVISION)
    tree.set_parent_trees([])
    tree.commit('revision 2')
    tree.branch.repository.copy_content_into(shared_repo)
    if shared_repo.make_working_trees():
        shared_repo.set_make_working_trees(False)
        self.assertFalse(shared_repo.make_working_trees())
    self.assertTrue(shared_repo.has_revision(rev1))
    dir = self.make_controldir('shared/source')
    dir.create_branch()
    target = dir.sprout(self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertNotEqual(dir.transport.base, shared_repo.controldir.transport.base)
    branch = target.open_branch()
    self.assertEqual([], branch.repository.all_revision_ids())
    if branch.controldir._format.supports_workingtrees:
        self.assertTrue(branch.repository.make_working_trees())
    self.assertFalse(branch.repository.is_shared())