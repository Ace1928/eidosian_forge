import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
class CaptureList(run.TestToolsTestRunner):

    def list(self, test, loader=None):
        tests.append({case.id() for case in testtools.testsuite.iterate_tests(test)})
        tests.append(loader)