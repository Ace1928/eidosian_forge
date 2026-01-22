import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class TestCleanUp(unittest.TestCase):

    def testCleanUp(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        test = TestableTest('testNothing')
        self.assertEqual(test._cleanups, [])
        cleanups = []

        def cleanup1(*args, **kwargs):
            cleanups.append((1, args, kwargs))

        def cleanup2(*args, **kwargs):
            cleanups.append((2, args, kwargs))
        test.addCleanup(cleanup1, 1, 2, 3, four='hello', five='goodbye')
        test.addCleanup(cleanup2)
        self.assertEqual(test._cleanups, [(cleanup1, (1, 2, 3), dict(four='hello', five='goodbye')), (cleanup2, (), {})])
        self.assertTrue(test.doCleanups())
        self.assertEqual(cleanups, [(2, (), {}), (1, (1, 2, 3), dict(four='hello', five='goodbye'))])

    def testCleanUpWithErrors(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        test = TestableTest('testNothing')
        result = unittest.TestResult()
        outcome = test._outcome = _Outcome(result=result)
        CleanUpExc = CustomError('foo')
        exc2 = CustomError('bar')

        def cleanup1():
            raise CleanUpExc

        def cleanup2():
            raise exc2
        test.addCleanup(cleanup1)
        test.addCleanup(cleanup2)
        self.assertFalse(test.doCleanups())
        self.assertFalse(outcome.success)
        (_, msg2), (_, msg1) = result.errors
        self.assertIn('in cleanup1', msg1)
        self.assertIn('raise CleanUpExc', msg1)
        self.assertIn(f'{CustomErrorRepr}: foo', msg1)
        self.assertIn('in cleanup2', msg2)
        self.assertIn('raise exc2', msg2)
        self.assertIn(f'{CustomErrorRepr}: bar', msg2)

    def testCleanupInRun(self):
        blowUp = False
        ordering = []

        class TestableTest(unittest.TestCase):

            def setUp(self):
                ordering.append('setUp')
                test.addCleanup(cleanup2)
                if blowUp:
                    raise CustomError('foo')

            def testNothing(self):
                ordering.append('test')
                test.addCleanup(cleanup3)

            def tearDown(self):
                ordering.append('tearDown')
        test = TestableTest('testNothing')

        def cleanup1():
            ordering.append('cleanup1')

        def cleanup2():
            ordering.append('cleanup2')

        def cleanup3():
            ordering.append('cleanup3')
        test.addCleanup(cleanup1)

        def success(some_test):
            self.assertEqual(some_test, test)
            ordering.append('success')
        result = unittest.TestResult()
        result.addSuccess = success
        test.run(result)
        self.assertEqual(ordering, ['setUp', 'test', 'tearDown', 'cleanup3', 'cleanup2', 'cleanup1', 'success'])
        blowUp = True
        ordering = []
        test = TestableTest('testNothing')
        test.addCleanup(cleanup1)
        test.run(result)
        self.assertEqual(ordering, ['setUp', 'cleanup2', 'cleanup1'])

    def testTestCaseDebugExecutesCleanups(self):
        ordering = []

        class TestableTest(unittest.TestCase):

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup1)

            def testNothing(self):
                ordering.append('test')
                self.addCleanup(cleanup3)

            def tearDown(self):
                ordering.append('tearDown')
                test.addCleanup(cleanup4)
        test = TestableTest('testNothing')

        def cleanup1():
            ordering.append('cleanup1')
            test.addCleanup(cleanup2)

        def cleanup2():
            ordering.append('cleanup2')

        def cleanup3():
            ordering.append('cleanup3')

        def cleanup4():
            ordering.append('cleanup4')
        test.debug()
        self.assertEqual(ordering, ['setUp', 'test', 'tearDown', 'cleanup4', 'cleanup3', 'cleanup1', 'cleanup2'])

    def test_enterContext(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        test = TestableTest('testNothing')
        cleanups = []
        test.addCleanup(cleanups.append, 'cleanup1')
        cm = TestCM(cleanups, 42)
        self.assertEqual(test.enterContext(cm), 42)
        test.addCleanup(cleanups.append, 'cleanup2')
        self.assertTrue(test.doCleanups())
        self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])

    def test_enterContext_arg_errors(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        test = TestableTest('testNothing')
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            test.enterContext(LacksEnterAndExit())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            test.enterContext(LacksEnter())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            test.enterContext(LacksExit())
        self.assertEqual(test._cleanups, [])