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
@decorator
class SkippingTestCase(TestCase):
    setup_ran = False

    def setUp(self):
        super().setUp()
        self.setup_ran = True

    def test_skipped(self):
        self.fail()