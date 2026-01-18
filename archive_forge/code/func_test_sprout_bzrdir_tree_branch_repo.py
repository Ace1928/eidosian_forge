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
def test_sprout_bzrdir_tree_branch_repo(self):
    tree = self.make_branch_and_tree('source')
    self.build_tree(['foo'], transport=tree.controldir.transport.clone('..'))
    tree.add('foo')
    tree.commit('revision 1')
    dir = tree.controldir
    target = self.sproutOrSkip(dir, self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport, ['./.bzr/branch', './.bzr/checkout/dirstate', './.bzr/checkout/stat-cache', './.bzr/checkout/inventory', './.bzr/inventory', './.bzr/parent', './.bzr/repository', './.bzr/stat-cache'])
    self.assertRepositoryHasSameItems(tree.branch.repository, target.open_repository())