import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_not_branch_bzrdir_with_repo(self):
    controldir = self.make_repository('repo').controldir
    err = errors.NotBranchError('path', controldir=controldir)
    self.assertEqual('Not a branch: "path": location is a repository.', str(err))