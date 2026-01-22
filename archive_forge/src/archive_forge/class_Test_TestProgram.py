import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
class Test_TestProgram(unittest.TestCase):

    def test_discovery_from_dotted_path(self):
        loader = unittest.TestLoader()
        tests = [self]
        expectedPath = os.path.abspath(os.path.dirname(unittest.test.__file__))
        self.wasRun = False

        def _find_tests(start_dir, pattern):
            self.wasRun = True
            self.assertEqual(start_dir, expectedPath)
            return tests
        loader._find_tests = _find_tests
        suite = loader.discover('unittest.test')
        self.assertTrue(self.wasRun)
        self.assertEqual(suite._tests, tests)

    def testNoExit(self):
        result = object()
        test = object()

        class FakeRunner(object):

            def run(self, test):
                self.test = test
                return result
        runner = FakeRunner()
        oldParseArgs = unittest.TestProgram.parseArgs

        def restoreParseArgs():
            unittest.TestProgram.parseArgs = oldParseArgs
        unittest.TestProgram.parseArgs = lambda *args: None
        self.addCleanup(restoreParseArgs)

        def removeTest():
            del unittest.TestProgram.test
        unittest.TestProgram.test = test
        self.addCleanup(removeTest)
        program = unittest.TestProgram(testRunner=runner, exit=False, verbosity=2)
        self.assertEqual(program.result, result)
        self.assertEqual(runner.test, test)
        self.assertEqual(program.verbosity, 2)

    class FooBar(unittest.TestCase):

        def testPass(self):
            pass

        def testFail(self):
            raise AssertionError

        def testError(self):
            1 / 0

        @unittest.skip('skipping')
        def testSkipped(self):
            raise AssertionError

        @unittest.expectedFailure
        def testExpectedFailure(self):
            raise AssertionError

        @unittest.expectedFailure
        def testUnexpectedSuccess(self):
            pass

    class FooBarLoader(unittest.TestLoader):
        """Test loader that returns a suite containing FooBar."""

        def loadTestsFromModule(self, module):
            return self.suiteClass([self.loadTestsFromTestCase(Test_TestProgram.FooBar)])

        def loadTestsFromNames(self, names, module):
            return self.suiteClass([self.loadTestsFromTestCase(Test_TestProgram.FooBar)])

    def test_defaultTest_with_string(self):

        class FakeRunner(object):

            def run(self, test):
                self.test = test
                return True
        old_argv = sys.argv
        sys.argv = ['faketest']
        runner = FakeRunner()
        program = unittest.TestProgram(testRunner=runner, exit=False, defaultTest='unittest.test', testLoader=self.FooBarLoader())
        sys.argv = old_argv
        self.assertEqual(('unittest.test',), program.testNames)

    def test_defaultTest_with_iterable(self):

        class FakeRunner(object):

            def run(self, test):
                self.test = test
                return True
        old_argv = sys.argv
        sys.argv = ['faketest']
        runner = FakeRunner()
        program = unittest.TestProgram(testRunner=runner, exit=False, defaultTest=['unittest.test', 'unittest.test2'], testLoader=self.FooBarLoader())
        sys.argv = old_argv
        self.assertEqual(['unittest.test', 'unittest.test2'], program.testNames)

    def test_NonExit(self):
        stream = BufferedWriter()
        program = unittest.main(exit=False, argv=['foobar'], testRunner=unittest.TextTestRunner(stream=stream), testLoader=self.FooBarLoader())
        self.assertTrue(hasattr(program, 'result'))
        out = stream.getvalue()
        self.assertIn('\nFAIL: testFail ', out)
        self.assertIn('\nERROR: testError ', out)
        self.assertIn('\nUNEXPECTED SUCCESS: testUnexpectedSuccess ', out)
        expected = '\n\nFAILED (failures=1, errors=1, skipped=1, expected failures=1, unexpected successes=1)\n'
        self.assertTrue(out.endswith(expected))

    def test_Exit(self):
        stream = BufferedWriter()
        self.assertRaises(SystemExit, unittest.main, argv=['foobar'], testRunner=unittest.TextTestRunner(stream=stream), exit=True, testLoader=self.FooBarLoader())
        out = stream.getvalue()
        self.assertIn('\nFAIL: testFail ', out)
        self.assertIn('\nERROR: testError ', out)
        self.assertIn('\nUNEXPECTED SUCCESS: testUnexpectedSuccess ', out)
        expected = '\n\nFAILED (failures=1, errors=1, skipped=1, expected failures=1, unexpected successes=1)\n'
        self.assertTrue(out.endswith(expected))

    def test_ExitAsDefault(self):
        stream = BufferedWriter()
        self.assertRaises(SystemExit, unittest.main, argv=['foobar'], testRunner=unittest.TextTestRunner(stream=stream), testLoader=self.FooBarLoader())
        out = stream.getvalue()
        self.assertIn('\nFAIL: testFail ', out)
        self.assertIn('\nERROR: testError ', out)
        self.assertIn('\nUNEXPECTED SUCCESS: testUnexpectedSuccess ', out)
        expected = '\n\nFAILED (failures=1, errors=1, skipped=1, expected failures=1, unexpected successes=1)\n'
        self.assertTrue(out.endswith(expected))