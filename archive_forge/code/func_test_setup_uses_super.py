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
def test_setup_uses_super(self):

    class OtherBaseCase(unittest.TestCase):
        setup_called = False

        def setUp(self):
            self.setup_called = True
            super().setUp()

    class OurCase(TestCase, OtherBaseCase):

        def runTest(self):
            pass
    test = OurCase()
    test.setUp()
    test.tearDown()
    self.assertTrue(test.setup_called)