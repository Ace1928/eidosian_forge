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
def test_sprout_controldir_repository_revision(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    br = tree.controldir.open_branch()
    br.set_last_revision_info(0, _mod_revision.NULL_REVISION)
    tree.set_parent_trees([])
    rev2 = tree.commit('revision 2')
    source = self.make_repository('source')
    tree.branch.repository.copy_content_into(source)
    dir = source.controldir
    self.sproutOrSkip(dir, self.get_url('target'), revision_id=rev2)
    raise TestSkipped('revision limiting not strict yet')