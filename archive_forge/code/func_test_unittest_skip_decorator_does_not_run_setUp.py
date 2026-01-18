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
def test_unittest_skip_decorator_does_not_run_setUp(self):
    reason = self.getUniqueString()
    self.check_skip_decorator_does_not_run_setup(unittest.skip(reason), reason)