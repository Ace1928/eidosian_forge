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
def test_same_lockfiles_between_tree_repo_branch(self):
    dir = BzrDirFormat6().initialize(self.get_url())

    def check_dir_components_use_same_lock(dir):
        ctrl_1 = dir.open_repository().control_files
        ctrl_2 = dir.open_branch().control_files
        ctrl_3 = dir.open_workingtree()._control_files
        self.assertTrue(ctrl_1 is ctrl_2)
        self.assertTrue(ctrl_2 is ctrl_3)
    check_dir_components_use_same_lock(dir)
    dir = controldir.ControlDir.open(self.get_url())
    check_dir_components_use_same_lock(dir)