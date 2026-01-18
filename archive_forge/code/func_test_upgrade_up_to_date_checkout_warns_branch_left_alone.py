import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_upgrade_up_to_date_checkout_warns_branch_left_alone(self):
    self.make_current_format_branch_and_checkout()
    burl = self.get_transport('current_format_branch').local_abspath('.')
    curl = self.get_transport('current_format_checkout').local_abspath('.')
    out, err = self.run_bzr('upgrade current_format_checkout', retcode=0)
    self.assertEqual('Upgrading branch %s/ ...\nThis is a checkout. The branch (%s/) needs to be upgraded separately.\nThe branch format %s is already at the most recent format.\n' % (curl, burl, 'Meta directory format 1'), out)