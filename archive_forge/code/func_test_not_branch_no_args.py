import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_not_branch_no_args(self):
    err = errors.NotBranchError('path')
    self.assertEqual('Not a branch: "path".', str(err))