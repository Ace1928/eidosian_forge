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
def test_run_load_list(self):
    self.useFixture(SampleTestFixture())
    out = io.StringIO()
    tempdir = self.useFixture(fixtures.TempDir())
    tempname = tempdir.path + '/tests.list'
    f = open(tempname, 'wb')
    try:
        f.write(_b('\ntesttools.runexample.TestFoo.test_bar\ntesttools.runexample.missingtest\n'))
    finally:
        f.close()
    try:
        run.main(['prog', '-l', '--load-list', tempname, 'testtools.runexample.test_suite'], out)
    except SystemExit:
        exc_info = sys.exc_info()
        raise AssertionError('-l --load-list tried to exit. %r' % exc_info[1])
    self.assertEqual('testtools.runexample.TestFoo.test_bar\n', out.getvalue())