import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class TestOutputBuffering(unittest.TestCase):

    def setUp(self):
        self._real_out = sys.stdout
        self._real_err = sys.stderr

    def tearDown(self):
        sys.stdout = self._real_out
        sys.stderr = self._real_err

    def testBufferOutputOff(self):
        real_out = self._real_out
        real_err = self._real_err
        result = unittest.TestResult()
        self.assertFalse(result.buffer)
        self.assertIs(real_out, sys.stdout)
        self.assertIs(real_err, sys.stderr)
        result.startTest(self)
        self.assertIs(real_out, sys.stdout)
        self.assertIs(real_err, sys.stderr)

    def testBufferOutputStartTestAddSuccess(self):
        real_out = self._real_out
        real_err = self._real_err
        result = unittest.TestResult()
        self.assertFalse(result.buffer)
        result.buffer = True
        self.assertIs(real_out, sys.stdout)
        self.assertIs(real_err, sys.stderr)
        result.startTest(self)
        self.assertIsNot(real_out, sys.stdout)
        self.assertIsNot(real_err, sys.stderr)
        self.assertIsInstance(sys.stdout, io.StringIO)
        self.assertIsInstance(sys.stderr, io.StringIO)
        self.assertIsNot(sys.stdout, sys.stderr)
        out_stream = sys.stdout
        err_stream = sys.stderr
        result._original_stdout = io.StringIO()
        result._original_stderr = io.StringIO()
        print('foo')
        print('bar', file=sys.stderr)
        self.assertEqual(out_stream.getvalue(), 'foo\n')
        self.assertEqual(err_stream.getvalue(), 'bar\n')
        self.assertEqual(result._original_stdout.getvalue(), '')
        self.assertEqual(result._original_stderr.getvalue(), '')
        result.addSuccess(self)
        result.stopTest(self)
        self.assertIs(sys.stdout, result._original_stdout)
        self.assertIs(sys.stderr, result._original_stderr)
        self.assertEqual(result._original_stdout.getvalue(), '')
        self.assertEqual(result._original_stderr.getvalue(), '')
        self.assertEqual(out_stream.getvalue(), '')
        self.assertEqual(err_stream.getvalue(), '')

    def getStartedResult(self):
        result = unittest.TestResult()
        result.buffer = True
        result.startTest(self)
        return result

    def testBufferOutputAddErrorOrFailure(self):
        unittest.result.traceback = MockTraceback
        self.addCleanup(restore_traceback)
        for message_attr, add_attr, include_error in [('errors', 'addError', True), ('failures', 'addFailure', False), ('errors', 'addError', True), ('failures', 'addFailure', False)]:
            result = self.getStartedResult()
            buffered_out = sys.stdout
            buffered_err = sys.stderr
            result._original_stdout = io.StringIO()
            result._original_stderr = io.StringIO()
            print('foo', file=sys.stdout)
            if include_error:
                print('bar', file=sys.stderr)
            addFunction = getattr(result, add_attr)
            addFunction(self, (None, None, None))
            result.stopTest(self)
            result_list = getattr(result, message_attr)
            self.assertEqual(len(result_list), 1)
            test, message = result_list[0]
            expectedOutMessage = textwrap.dedent('\n                Stdout:\n                foo\n            ')
            expectedErrMessage = ''
            if include_error:
                expectedErrMessage = textwrap.dedent('\n                Stderr:\n                bar\n            ')
            expectedFullMessage = 'A traceback%s%s' % (expectedOutMessage, expectedErrMessage)
            self.assertIs(test, self)
            self.assertEqual(result._original_stdout.getvalue(), expectedOutMessage)
            self.assertEqual(result._original_stderr.getvalue(), expectedErrMessage)
            self.assertMultiLineEqual(message, expectedFullMessage)

    def testBufferSetUp(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def setUp(self):
                print('set up')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = f'test_foo ({strclass(Foo)}.test_foo)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(str(test_case), description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDown(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def tearDown(self):
                print('tear down')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = f'test_foo ({strclass(Foo)}.test_foo)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(str(test_case), description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferDoCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def setUp(self):
                print('set up')
                self.addCleanup(bad_cleanup1)
                self.addCleanup(bad_cleanup2)

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 2)
        description = f'test_foo ({strclass(Foo)}.test_foo)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(str(test_case), description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up\ndo cleanup2\n', formatted_exc)
        self.assertNotIn('\ndo cleanup1\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(str(test_case), description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferSetUp_DoCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def setUp(self):
                print('set up')
                self.addCleanup(bad_cleanup1)
                self.addCleanup(bad_cleanup2)
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 3)
        description = f'test_foo ({strclass(Foo)}.test_foo)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(str(test_case), description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up\n', formatted_exc)
        self.assertNotIn('\ndo cleanup2\n', formatted_exc)
        self.assertNotIn('\ndo cleanup1\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(str(test_case), description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up\ndo cleanup2\n', formatted_exc)
        self.assertNotIn('\ndo cleanup1\n', formatted_exc)
        test_case, formatted_exc = result.errors[2]
        self.assertEqual(str(test_case), description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDown_DoCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def setUp(self):
                print('set up')
                self.addCleanup(bad_cleanup1)
                self.addCleanup(bad_cleanup2)

            def tearDown(self):
                print('tear down')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up\ntear down\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 3)
        description = f'test_foo ({strclass(Foo)}.test_foo)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(str(test_case), description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up\ntear down\n', formatted_exc)
        self.assertNotIn('\ndo cleanup2\n', formatted_exc)
        self.assertNotIn('\ndo cleanup1\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(str(test_case), description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up\ntear down\ndo cleanup2\n', formatted_exc)
        self.assertNotIn('\ndo cleanup1\n', formatted_exc)
        test_case, formatted_exc = result.errors[2]
        self.assertEqual(str(test_case), description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferSetupClass(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                print('set up class')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up class\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = f'setUpClass ({strclass(Foo)})'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDownClass(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            @classmethod
            def tearDownClass(cls):
                print('tear down class')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down class\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = f'tearDownClass ({strclass(Foo)})'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferDoClassCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                print('set up class')
                cls.addClassCleanup(bad_cleanup1)
                cls.addClassCleanup(bad_cleanup2)

            @classmethod
            def tearDownClass(cls):
                print('tear down class')

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down class\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 2)
        description = f'tearDownClass ({strclass(Foo)})'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(test_case.description, description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferSetupClass_DoClassCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                print('set up class')
                cls.addClassCleanup(bad_cleanup1)
                cls.addClassCleanup(bad_cleanup2)
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up class\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 3)
        description = f'setUpClass ({strclass(Foo)})'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up class\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)
        test_case, formatted_exc = result.errors[2]
        self.assertEqual(test_case.description, description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDownClass_DoClassCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                print('set up class')
                cls.addClassCleanup(bad_cleanup1)
                cls.addClassCleanup(bad_cleanup2)

            @classmethod
            def tearDownClass(cls):
                print('tear down class')
                1 / 0

            def test_foo(self):
                pass
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down class\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 3)
        description = f'tearDownClass ({strclass(Foo)})'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\ntear down class\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)
        test_case, formatted_exc = result.errors[2]
        self.assertEqual(test_case.description, description)
        self.assertIn('TypeError: bad cleanup1', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferSetUpModule(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def test_foo(self):
                pass

        class Module(object):

            @staticmethod
            def setUpModule():
                print('set up module')
                1 / 0
        Foo.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up module\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = 'setUpModule (Module)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDownModule(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def test_foo(self):
                pass

        class Module(object):

            @staticmethod
            def tearDownModule():
                print('tear down module')
                1 / 0
        Foo.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down module\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = 'tearDownModule (Module)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferDoModuleCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def test_foo(self):
                pass

        class Module(object):

            @staticmethod
            def setUpModule():
                print('set up module')
                unittest.addModuleCleanup(bad_cleanup1)
                unittest.addModuleCleanup(bad_cleanup2)
        Foo.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 1)
        description = 'tearDownModule (Module)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferSetUpModule_DoModuleCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def test_foo(self):
                pass

        class Module(object):

            @staticmethod
            def setUpModule():
                print('set up module')
                unittest.addModuleCleanup(bad_cleanup1)
                unittest.addModuleCleanup(bad_cleanup2)
                1 / 0
        Foo.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\nset up module\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 2)
        description = 'setUpModule (Module)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\nset up module\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertIn(expected_out, formatted_exc)
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)

    def testBufferTearDownModule_DoModuleCleanups(self):
        with captured_stdout() as stdout:
            result = unittest.TestResult()
        result.buffer = True

        class Foo(unittest.TestCase):

            def test_foo(self):
                pass

        class Module(object):

            @staticmethod
            def setUpModule():
                print('set up module')
                unittest.addModuleCleanup(bad_cleanup1)
                unittest.addModuleCleanup(bad_cleanup2)

            @staticmethod
            def tearDownModule():
                print('tear down module')
                1 / 0
        Foo.__module__ = 'Module'
        sys.modules['Module'] = Module
        self.addCleanup(sys.modules.pop, 'Module')
        suite = unittest.TestSuite([Foo('test_foo')])
        suite(result)
        expected_out = '\nStdout:\ntear down module\ndo cleanup2\ndo cleanup1\n'
        self.assertEqual(stdout.getvalue(), expected_out)
        self.assertEqual(len(result.errors), 2)
        description = 'tearDownModule (Module)'
        test_case, formatted_exc = result.errors[0]
        self.assertEqual(test_case.description, description)
        self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
        self.assertNotIn('ValueError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn('\nStdout:\ntear down module\n', formatted_exc)
        test_case, formatted_exc = result.errors[1]
        self.assertEqual(test_case.description, description)
        self.assertIn('ValueError: bad cleanup2', formatted_exc)
        self.assertNotIn('ZeroDivisionError', formatted_exc)
        self.assertNotIn('TypeError', formatted_exc)
        self.assertIn(expected_out, formatted_exc)