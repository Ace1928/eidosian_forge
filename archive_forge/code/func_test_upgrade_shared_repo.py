import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_shared_repo(self):
    repo = self.make_repository('repo', format='2a', shared=True)
    branch = self.make_branch_and_tree('repo/branch', format='pack-0.92')
    self.get_transport('repo/branch/.bzr/repository').delete_tree('.')
    out, err = self.run_bzr(['upgrade'], working_dir='repo/branch')