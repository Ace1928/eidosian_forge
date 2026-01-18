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
def test_revert_inventory(self):
    tree = self.make_branch_and_tree('source')
    self.build_tree(['source/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    dir = tree.controldir
    target = dir.clone(self.get_url('target'))
    self.skipIfNoWorkingTree(target)
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport, ['./.bzr/stat-cache', './.bzr/checkout/dirstate', './.bzr/checkout/stat-cache', './.bzr/checkout/merge-hashes', './.bzr/merge-hashes', './.bzr/repository'])
    self.assertRepositoryHasSameItems(tree.branch.repository, target.open_branch().repository)
    target.open_workingtree().revert()
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport, ['./.bzr/stat-cache', './.bzr/checkout/dirstate', './.bzr/checkout/stat-cache', './.bzr/checkout/merge-hashes', './.bzr/merge-hashes', './.bzr/repository'])
    self.assertRepositoryHasSameItems(tree.branch.repository, target.open_branch().repository)