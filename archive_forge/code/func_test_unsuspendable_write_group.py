import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_unsuspendable_write_group(self):
    repo = 'dummy repo'
    err = errors.UnsuspendableWriteGroup(repo)
    self.assertEqual('Repository dummy repo cannot suspend a write group.', str(err))