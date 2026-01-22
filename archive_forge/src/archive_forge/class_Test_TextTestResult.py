import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class Test_TextTestResult(unittest.TestCase):
    maxDiff = None

    def testGetDescriptionWithoutDocstring(self):
        result = unittest.TextTestResult(None, True, 1)
        self.assertEqual(result.getDescription(self), 'testGetDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetDescriptionWithoutDocstring)')

    def testGetSubTestDescriptionWithoutDocstring(self):
        with self.subTest(foo=1, bar=2):
            result = unittest.TextTestResult(None, True, 1)
            self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithoutDocstring) (foo=1, bar=2)')
        with self.subTest('some message'):
            result = unittest.TextTestResult(None, True, 1)
            self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithoutDocstring) [some message]')

    def testGetSubTestDescriptionWithoutDocstringAndParams(self):
        with self.subTest():
            result = unittest.TextTestResult(None, True, 1)
            self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithoutDocstringAndParams (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithoutDocstringAndParams) (<subtest>)')

    def testGetSubTestDescriptionForFalsyValues(self):
        expected = 'testGetSubTestDescriptionForFalsyValues (%s.Test_TextTestResult.testGetSubTestDescriptionForFalsyValues) [%s]'
        result = unittest.TextTestResult(None, True, 1)
        for arg in [0, None, []]:
            with self.subTest(arg):
                self.assertEqual(result.getDescription(self._subtest), expected % (__name__, arg))

    def testGetNestedSubTestDescriptionWithoutDocstring(self):
        with self.subTest(foo=1):
            with self.subTest(baz=2, bar=3):
                result = unittest.TextTestResult(None, True, 1)
                self.assertEqual(result.getDescription(self._subtest), 'testGetNestedSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetNestedSubTestDescriptionWithoutDocstring) (baz=2, bar=3, foo=1)')

    def testGetDuplicatedNestedSubTestDescriptionWithoutDocstring(self):
        with self.subTest(foo=1, bar=2):
            with self.subTest(baz=3, bar=4):
                result = unittest.TextTestResult(None, True, 1)
                self.assertEqual(result.getDescription(self._subtest), 'testGetDuplicatedNestedSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetDuplicatedNestedSubTestDescriptionWithoutDocstring) (baz=3, bar=4, foo=1)')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def testGetDescriptionWithOneLineDocstring(self):
        """Tests getDescription() for a method with a docstring."""
        result = unittest.TextTestResult(None, True, 1)
        self.assertEqual(result.getDescription(self), 'testGetDescriptionWithOneLineDocstring (' + __name__ + '.Test_TextTestResult.testGetDescriptionWithOneLineDocstring)\nTests getDescription() for a method with a docstring.')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def testGetSubTestDescriptionWithOneLineDocstring(self):
        """Tests getDescription() for a method with a docstring."""
        result = unittest.TextTestResult(None, True, 1)
        with self.subTest(foo=1, bar=2):
            self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithOneLineDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithOneLineDocstring) (foo=1, bar=2)\nTests getDescription() for a method with a docstring.')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def testGetDescriptionWithMultiLineDocstring(self):
        """Tests getDescription() for a method with a longer docstring.
        The second line of the docstring.
        """
        result = unittest.TextTestResult(None, True, 1)
        self.assertEqual(result.getDescription(self), 'testGetDescriptionWithMultiLineDocstring (' + __name__ + '.Test_TextTestResult.testGetDescriptionWithMultiLineDocstring)\nTests getDescription() for a method with a longer docstring.')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def testGetSubTestDescriptionWithMultiLineDocstring(self):
        """Tests getDescription() for a method with a longer docstring.
        The second line of the docstring.
        """
        result = unittest.TextTestResult(None, True, 1)
        with self.subTest(foo=1, bar=2):
            self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithMultiLineDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithMultiLineDocstring) (foo=1, bar=2)\nTests getDescription() for a method with a longer docstring.')

    class Test(unittest.TestCase):

        def testSuccess(self):
            pass

        def testSkip(self):
            self.skipTest('skip')

        def testFail(self):
            self.fail('fail')

        def testError(self):
            raise Exception('error')

        @unittest.expectedFailure
        def testExpectedFailure(self):
            self.fail('fail')

        @unittest.expectedFailure
        def testUnexpectedSuccess(self):
            pass

        def testSubTestSuccess(self):
            with self.subTest('one', a=1):
                pass
            with self.subTest('two', b=2):
                pass

        def testSubTestMixed(self):
            with self.subTest('success', a=1):
                pass
            with self.subTest('skip', b=2):
                self.skipTest('skip')
            with self.subTest('fail', c=3):
                self.fail('fail')
            with self.subTest('error', d=4):
                raise Exception('error')
        tearDownError = None

        def tearDown(self):
            if self.tearDownError is not None:
                raise self.tearDownError

    def _run_test(self, test_name, verbosity, tearDownError=None):
        stream = BufferedWriter()
        stream = unittest.runner._WritelnDecorator(stream)
        result = unittest.TextTestResult(stream, True, verbosity)
        test = self.Test(test_name)
        test.tearDownError = tearDownError
        test.run(result)
        return stream.getvalue()

    def testDotsOutput(self):
        self.assertEqual(self._run_test('testSuccess', 1), '.')
        self.assertEqual(self._run_test('testSkip', 1), 's')
        self.assertEqual(self._run_test('testFail', 1), 'F')
        self.assertEqual(self._run_test('testError', 1), 'E')
        self.assertEqual(self._run_test('testExpectedFailure', 1), 'x')
        self.assertEqual(self._run_test('testUnexpectedSuccess', 1), 'u')

    def testLongOutput(self):
        classname = f'{__name__}.{self.Test.__qualname__}'
        self.assertEqual(self._run_test('testSuccess', 2), f'testSuccess ({classname}.testSuccess) ... ok\n')
        self.assertEqual(self._run_test('testSkip', 2), f"testSkip ({classname}.testSkip) ... skipped 'skip'\n")
        self.assertEqual(self._run_test('testFail', 2), f'testFail ({classname}.testFail) ... FAIL\n')
        self.assertEqual(self._run_test('testError', 2), f'testError ({classname}.testError) ... ERROR\n')
        self.assertEqual(self._run_test('testExpectedFailure', 2), f'testExpectedFailure ({classname}.testExpectedFailure) ... expected failure\n')
        self.assertEqual(self._run_test('testUnexpectedSuccess', 2), f'testUnexpectedSuccess ({classname}.testUnexpectedSuccess) ... unexpected success\n')

    def testDotsOutputSubTestSuccess(self):
        self.assertEqual(self._run_test('testSubTestSuccess', 1), '.')

    def testLongOutputSubTestSuccess(self):
        classname = f'{__name__}.{self.Test.__qualname__}'
        self.assertEqual(self._run_test('testSubTestSuccess', 2), f'testSubTestSuccess ({classname}.testSubTestSuccess) ... ok\n')

    def testDotsOutputSubTestMixed(self):
        self.assertEqual(self._run_test('testSubTestMixed', 1), 'sFE')

    def testLongOutputSubTestMixed(self):
        classname = f'{__name__}.{self.Test.__qualname__}'
        self.assertEqual(self._run_test('testSubTestMixed', 2), f"testSubTestMixed ({classname}.testSubTestMixed) ... \n  testSubTestMixed ({classname}.testSubTestMixed) [skip] (b=2) ... skipped 'skip'\n  testSubTestMixed ({classname}.testSubTestMixed) [fail] (c=3) ... FAIL\n  testSubTestMixed ({classname}.testSubTestMixed) [error] (d=4) ... ERROR\n")

    def testDotsOutputTearDownFail(self):
        out = self._run_test('testSuccess', 1, AssertionError('fail'))
        self.assertEqual(out, 'F')
        out = self._run_test('testError', 1, AssertionError('fail'))
        self.assertEqual(out, 'EF')
        out = self._run_test('testFail', 1, Exception('error'))
        self.assertEqual(out, 'FE')
        out = self._run_test('testSkip', 1, AssertionError('fail'))
        self.assertEqual(out, 'sF')

    def testLongOutputTearDownFail(self):
        classname = f'{__name__}.{self.Test.__qualname__}'
        out = self._run_test('testSuccess', 2, AssertionError('fail'))
        self.assertEqual(out, f'testSuccess ({classname}.testSuccess) ... FAIL\n')
        out = self._run_test('testError', 2, AssertionError('fail'))
        self.assertEqual(out, f'testError ({classname}.testError) ... ERROR\ntestError ({classname}.testError) ... FAIL\n')
        out = self._run_test('testFail', 2, Exception('error'))
        self.assertEqual(out, f'testFail ({classname}.testFail) ... FAIL\ntestFail ({classname}.testFail) ... ERROR\n')
        out = self._run_test('testSkip', 2, AssertionError('fail'))
        self.assertEqual(out, f"testSkip ({classname}.testSkip) ... skipped 'skip'\ntestSkip ({classname}.testSkip) ... FAIL\n")