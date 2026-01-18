import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_clean_supported(self):
    self.assertLegalOption('--clean')
    self.assertBranchFormat('branch-foo', '2a')
    backup_bzr_dir = os.path.join('branch-foo', 'backup.bzr.~1~')
    self.assertFalse(os.path.exists(backup_bzr_dir))