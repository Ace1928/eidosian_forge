import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_not_write_locked(self):
    error = errors.NotWriteLocked('a thing to repr')
    self.assertEqualDiff("'a thing to repr' is not write locked but needs to be.", str(error))