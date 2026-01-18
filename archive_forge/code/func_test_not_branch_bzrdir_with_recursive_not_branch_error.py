import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_not_branch_bzrdir_with_recursive_not_branch_error(self):

    class FakeBzrDir:

        def open_repository(self):
            raise errors.NotBranchError('path', controldir=FakeBzrDir())
    err = errors.NotBranchError('path', controldir=FakeBzrDir())
    self.assertEqual('Not a branch: "path": NotBranchError.', str(err))