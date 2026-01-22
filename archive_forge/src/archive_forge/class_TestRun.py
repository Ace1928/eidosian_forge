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
class TestRun(TestCase):

    def setUp(self):
        super().setUp()
        if fixtures is None:
            self.skipTest('Need fixtures')

    def test_run_custom_list(self):
        self.useFixture(SampleTestFixture())
        tests = []

        class CaptureList(run.TestToolsTestRunner):

            def list(self, test):
                tests.append({case.id() for case in testtools.testsuite.iterate_tests(test)})
        out = io.StringIO()
        try:
            program = run.TestProgram(argv=['prog', '-l', 'testtools.runexample.test_suite'], stdout=out, testRunner=CaptureList)
        except SystemExit:
            exc_info = sys.exc_info()
            raise AssertionError('-l tried to exit. %r' % exc_info[1])
        self.assertEqual([{'testtools.runexample.TestFoo.test_bar', 'testtools.runexample.TestFoo.test_quux'}], tests)

    def test_run_list_with_loader(self):
        self.useFixture(SampleTestFixture())
        tests = []

        class CaptureList(run.TestToolsTestRunner):

            def list(self, test, loader=None):
                tests.append({case.id() for case in testtools.testsuite.iterate_tests(test)})
                tests.append(loader)
        out = io.StringIO()
        try:
            program = run.TestProgram(argv=['prog', '-l', 'testtools.runexample.test_suite'], stdout=out, testRunner=CaptureList)
        except SystemExit:
            exc_info = sys.exc_info()
            raise AssertionError('-l tried to exit. %r' % exc_info[1])
        self.assertEqual([{'testtools.runexample.TestFoo.test_bar', 'testtools.runexample.TestFoo.test_quux'}, program.testLoader], tests)

    def test_run_list(self):
        self.useFixture(SampleTestFixture())
        out = io.StringIO()
        try:
            run.main(['prog', '-l', 'testtools.runexample.test_suite'], out)
        except SystemExit:
            exc_info = sys.exc_info()
            raise AssertionError('-l tried to exit. %r' % exc_info[1])
        self.assertEqual('testtools.runexample.TestFoo.test_bar\ntesttools.runexample.TestFoo.test_quux\n', out.getvalue())

    def test_run_list_failed_import(self):
        broken = self.useFixture(SampleTestFixture(broken=True))
        out = io.StringIO()
        unittest.defaultTestLoader._top_level_dir = None
        exc = self.assertRaises(SystemExit, run.main, ['prog', 'discover', '-l', broken.package.base, '*.py'], out)
        self.assertEqual(2, exc.args[0])
        self.assertThat(out.getvalue(), DocTestMatches('unittest.loader._FailedTest.runexample\nFailed to import test module: runexample\nTraceback (most recent call last):\n  File ".../loader.py", line ..., in _find_test_path\n    package = self._get_module_from_name(name)...\n  File ".../loader.py", line ..., in _get_module_from_name\n    __import__(name)...\n  File ".../runexample/__init__.py", line 1\n    class not in\n...^...\nSyntaxError: invalid syntax\n\n', doctest.ELLIPSIS))

    def test_run_orders_tests(self):
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

    def test_load_list_preserves_custom_suites(self):
        if testresources is None:
            self.skipTest('Need testresources')
        self.useFixture(SampleResourcedFixture())
        tempdir = self.useFixture(fixtures.TempDir())
        tempname = tempdir.path + '/tests.list'
        f = open(tempname, 'wb')
        try:
            f.write(_b('\ntesttools.resourceexample.TestFoo.test_bar\ntesttools.resourceexample.TestFoo.test_foo\n'))
        finally:
            f.close()
        stdout = self.useFixture(fixtures.StringStream('stdout'))
        with fixtures.MonkeyPatch('sys.stdout', stdout.stream):
            try:
                run.main(['prog', '--load-list', tempname, 'testtools.resourceexample.test_suite'], stdout.stream)
            except SystemExit:
                pass
        out = stdout.getDetails()['stdout'].as_text()
        self.assertEqual(1, out.count('Setting up Printer'), '%r' % out)

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

    def test_run_locals(self):
        stdout = self.useFixture(fixtures.StringStream('stdout'))

        class Failing(TestCase):

            def test_a(self):
                a = 1
                self.fail('a')
        runner = run.TestToolsTestRunner(tb_locals=True, stdout=stdout.stream)
        runner.run(Failing('test_a'))
        self.assertThat(stdout.getDetails()['stdout'].as_text(), Contains('a = 1'))

    def test_stdout_honoured(self):
        self.useFixture(SampleTestFixture())
        tests = []
        out = io.StringIO()
        exc = self.assertRaises(SystemExit, run.main, argv=['prog', 'testtools.runexample.test_suite'], stdout=out)
        self.assertEqual((0,), exc.args)
        self.assertThat(out.getvalue(), MatchesRegex('Tests running...\n\nRan 2 tests in \\d.\\d\\d\\ds\nOK\n'))

    @skipUnless(fixtures, 'fixtures not present')
    def test_issue_16662(self):
        pkg = self.useFixture(SampleLoadTestsPackage())
        out = io.StringIO()
        unittest.defaultTestLoader._top_level_dir = None
        self.assertEqual(None, run.main(['prog', 'discover', '-l', pkg.package.base], out))
        self.assertEqual(dedent('            discoverexample.TestExample.test_foo\n            fred\n            '), out.getvalue())