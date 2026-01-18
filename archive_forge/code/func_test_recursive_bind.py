import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_recursive_bind(self):
    error = errors.RecursiveBind('foo_bar_branch')
    msg = 'Branch "foo_bar_branch" appears to be bound to itself. Please use `brz unbind` to fix.'
    self.assertEqualDiff(msg, str(error))