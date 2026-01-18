import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_explicit_knit(self):
    self.make_branch_and_tree('branch', format='knit')
    transport = self.get_transport('branch')
    url = transport.base
    display_url = transport.local_abspath('.')
    backup_dir = 'backup.bzr.~1~'
    out, err = self.run_bzr(['upgrade', '--format=pack-0.92', url])
    self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting repository conversion\nrepository converted\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir), out)
    self.assertEqualDiff('', err)
    converted_dir = controldir.ControlDir.open(self.get_url('branch'))
    self.assertTrue(isinstance(converted_dir._format, bzrdir.BzrDirMetaFormat1))
    self.assertTrue(isinstance(converted_dir.open_repository()._format, RepositoryFormatKnitPack1))