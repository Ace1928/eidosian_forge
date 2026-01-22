import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def assertBranchFormat(self, dir, format):
    branch = controldir.ControlDir.open_tree_or_branch(self.get_url(dir))[1]
    branch_format = branch._format
    meta_format = controldir.format_registry.make_controldir(format)
    expected_format = meta_format.get_branch_format()
    self.assertEqual(expected_format, branch_format)