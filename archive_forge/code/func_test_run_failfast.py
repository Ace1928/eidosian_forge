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
def test_run_failfast(self):
    stdout = self.useFixture(fixtures.StringStream('stdout'))

    class Failing(TestCase):

        def test_a(self):
            self.fail('a')

        def test_b(self):
            self.fail('b')
    with fixtures.MonkeyPatch('sys.stdout', stdout.stream):
        runner = run.TestToolsTestRunner(failfast=True)
        runner.run(TestSuite([Failing('test_a'), Failing('test_b')]))
    self.assertThat(stdout.getDetails()['stdout'].as_text(), Contains('Ran 1 test'))