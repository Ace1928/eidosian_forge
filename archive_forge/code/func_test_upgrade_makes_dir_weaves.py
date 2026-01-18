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
def test_upgrade_makes_dir_weaves(self):
    self.build_tree_contents(_upgrade_dir_template)
    old_repodir = controldir.ControlDir.open_unsupported('.')
    old_repo_format = old_repodir.open_repository()._format
    upgrade.upgrade('.')
    repo = repository.Repository.open('.')
    self.assertNotEqual(old_repo_format.__class__, repo._format.__class__)
    repo.lock_read()
    self.addCleanup(repo.unlock)
    text_keys = repo.texts.keys()
    dir_keys = [key for key in text_keys if key[0] == b'dir-20051005095101-da1441ea3fa6917a']
    self.assertNotEqual([], dir_keys)