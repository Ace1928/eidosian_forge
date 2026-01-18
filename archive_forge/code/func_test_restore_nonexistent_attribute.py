from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_restore_nonexistent_attribute(self):

    def test_body(case):
        case.patch(self, 'doesntexist', 'patched')
        return self.doesntexist
    self.run_test(test_body)
    marker = object()
    value = getattr(self, 'doesntexist', marker)
    self.assertIs(marker, value)