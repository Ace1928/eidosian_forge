import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_readonly_url_error(self):
    self.make_branch_and_tree('old_format_branch', format='knit')
    out, err = self.run_bzr(['upgrade', self.get_readonly_url('old_format_branch')], retcode=3)
    err_msg = 'Upgrade URL cannot work with readonly URLs.'
    self.assertEqualDiff('conversion error: %s\nbrz: ERROR: %s\n' % (err_msg, err_msg), err)