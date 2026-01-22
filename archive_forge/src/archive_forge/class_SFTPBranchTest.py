import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
class SFTPBranchTest(TestCaseWithSFTPServer):
    """Test some stuff when accessing a bzr Branch over sftp"""

    def test_lock_file(self):
        b = self.make_branch('', format=BzrDirFormat6())
        b = branch.Branch.open(self.get_url())
        self.assertPathExists('.bzr/')
        self.assertPathExists('.bzr/branch-format')
        self.assertPathExists('.bzr/branch-lock')
        self.assertPathDoesNotExist('.bzr/branch-lock.write-lock')
        b.lock_write()
        self.assertPathExists('.bzr/branch-lock.write-lock')
        b.unlock()
        self.assertPathDoesNotExist('.bzr/branch-lock.write-lock')