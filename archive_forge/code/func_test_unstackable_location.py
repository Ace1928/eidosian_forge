import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_unstackable_location(self):
    error = errors.UnstackableLocationError('foo', 'bar')
    self.assertEqualDiff("The branch 'foo' cannot be stacked on 'bar'.", str(error))