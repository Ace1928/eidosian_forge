import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_read_error(self):
    path = 'a path'
    error = errors.ReadError(path)
    self.assertContainsRe(str(error), "^Error reading from u?'a path'.$")