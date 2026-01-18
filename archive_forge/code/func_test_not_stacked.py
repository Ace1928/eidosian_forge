import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_not_stacked(self):
    error = errors.NotStacked('a branch')
    self.assertEqualDiff("The branch 'a branch' is not stacked.", str(error))