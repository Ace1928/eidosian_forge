import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class TestModuleCleanUp(unittest.TestCase):

    def test_add_and_do_ModuleCleanup(self):
        module_cleanups = []

        def module_cleanup1(*args, **kwargs):
            module_cleanups.append((3, args, kwargs))

        def module_cleanup2(*args, **kwargs):
            module_cleanups.append((4, args, kwargs))

        class Module(object):
            unittest.addModuleCleanup(module_cleanup1, 1, 2, 3, four='hello', five='goodbye')
            unittest.addModuleCleanup(module_cleanup2)
        self.assertEqual(unittest.case._module_cleanups, [(module_cleanup1, (1, 2, 3), dict(four='hello', five='goodbye')), (module_cleanup2, (), {})])
        unittest.case.doModuleCleanups()
        self.assertEqual(module_cleanups, [(4, (), {}), (3, (1, 2, 3), dict(four='hello', five='goodbye'))])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_doModuleCleanup_with_errors_in_addModuleCleanup(self):
        module_cleanups = []

        def module_cleanup_good(*args, **kwargs):
            module_cleanups.append((3, args, kwargs))

        def module_cleanup_bad(*args, **kwargs):
            raise CustomError('CleanUpExc')

        class Module(object):
            unittest.addModuleCleanup(module_cleanup_good, 1, 2, 3, four='hello', five='goodbye')
            unittest.addModuleCleanup(module_cleanup_bad)
        self.assertEqual(unittest.case._module_cleanups, [(module_cleanup_good, (1, 2, 3), dict(four='hello', five='goodbye')), (module_cleanup_bad, (), {})])
        with self.assertRaises(CustomError) as e:
            unittest.case.doModuleCleanups()
        self.assertEqual(str(e.exception), 'CleanUpExc')
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_addModuleCleanup_arg_errors(self):
        cleanups = []

        def cleanup(*args, **kwargs):
            cleanups.append((args, kwargs))

        class Module(object):
            unittest.addModuleCleanup(cleanup, 1, 2, function='hello')
            with self.assertRaises(TypeError):
                unittest.addModuleCleanup(function=cleanup, arg='hello')
            with self.assertRaises(TypeError):
                unittest.addModuleCleanup()
        unittest.case.doModuleCleanups()
        self.assertEqual(cleanups, [((1, 2), {'function': 'hello'})])

    def test_run_module_cleanUp(self):
        blowUp = True
        ordering = []

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)
                if blowUp:
                    raise CustomError('setUpModule Exc')

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        result = runTests(TestableTest)
        self.assertEqual(ordering, ['setUpModule', 'cleanup_good'])
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: setUpModule Exc')
        ordering = []
        blowUp = False
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good'])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_run_multiple_module_cleanUp(self):
        blowUp = True
        blowUp2 = False
        ordering = []

        class Module1(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)
                if blowUp:
                    raise CustomError()

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class Module2(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule2')
                unittest.addModuleCleanup(cleanup, ordering)
                if blowUp2:
                    raise CustomError()

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule2')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')

        class TestableTest2(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass2')

            def testNothing(self):
                ordering.append('test2')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass2')
        TestableTest.__module__ = 'Module1'
        sys.modules['Module1'] = Module1
        TestableTest2.__module__ = 'Module2'
        sys.modules['Module2'] = Module2
        runTests(TestableTest, TestableTest2)
        self.assertEqual(ordering, ['setUpModule', 'cleanup_good', 'setUpModule2', 'setUpClass2', 'test2', 'tearDownClass2', 'tearDownModule2', 'cleanup_good'])
        ordering = []
        blowUp = False
        blowUp2 = True
        runTests(TestableTest, TestableTest2)
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good', 'setUpModule2', 'cleanup_good'])
        ordering = []
        blowUp = False
        blowUp2 = False
        runTests(TestableTest, TestableTest2)
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good', 'setUpModule2', 'setUpClass2', 'test2', 'tearDownClass2', 'tearDownModule2', 'cleanup_good'])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_run_module_cleanUp_without_teardown(self):
        ordering = []

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        runTests(TestableTest)
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'cleanup_good'])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_run_module_cleanUp_when_teardown_exception(self):
        ordering = []

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')
                raise CustomError('CleanUpExc')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good'])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_debug_module_executes_cleanUp(self):
        ordering = []
        blowUp = False

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering, blowUp=blowUp)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        suite.debug()
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good'])
        self.assertEqual(unittest.case._module_cleanups, [])
        ordering = []
        blowUp = True
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])
        self.assertEqual(unittest.case._module_cleanups, [])

    def test_debug_module_cleanUp_when_teardown_exception(self):
        ordering = []
        blowUp = False

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering, blowUp=blowUp)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')
                raise CustomError('TearDownModuleExc')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'TearDownModuleExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule'])
        self.assertTrue(unittest.case._module_cleanups)
        unittest.case._module_cleanups.clear()
        ordering = []
        blowUp = True
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
        with self.assertRaises(CustomError) as cm:
            suite.debug()
        self.assertEqual(str(cm.exception), 'TearDownModuleExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule'])
        self.assertTrue(unittest.case._module_cleanups)
        unittest.case._module_cleanups.clear()

    def test_addClassCleanup_arg_errors(self):
        cleanups = []

        def cleanup(*args, **kwargs):
            cleanups.append((args, kwargs))

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                cls.addClassCleanup(cleanup, 1, 2, function=3, cls=4)
                with self.assertRaises(TypeError):
                    cls.addClassCleanup(function=cleanup, arg='hello')

            def testNothing(self):
                pass
        with self.assertRaises(TypeError):
            TestableTest.addClassCleanup()
        with self.assertRaises(TypeError):
            unittest.TestCase.addCleanup(cls=TestableTest(), function=cleanup)
        runTests(TestableTest)
        self.assertEqual(cleanups, [((1, 2), {'function': 3, 'cls': 4})])

    def test_addCleanup_arg_errors(self):
        cleanups = []

        def cleanup(*args, **kwargs):
            cleanups.append((args, kwargs))

        class TestableTest(unittest.TestCase):

            def setUp(self2):
                self2.addCleanup(cleanup, 1, 2, function=3, self=4)
                with self.assertRaises(TypeError):
                    self2.addCleanup(function=cleanup, arg='hello')

            def testNothing(self):
                pass
        with self.assertRaises(TypeError):
            TestableTest().addCleanup()
        with self.assertRaises(TypeError):
            unittest.TestCase.addCleanup(self=TestableTest(), function=cleanup)
        runTests(TestableTest)
        self.assertEqual(cleanups, [((1, 2), {'function': 3, 'self': 4})])

    def test_with_errors_in_addClassCleanup(self):
        ordering = []

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                cls.addClassCleanup(cleanup, ordering, blowUp=True)

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'cleanup_exc', 'tearDownModule', 'cleanup_good'])

    def test_with_errors_in_addCleanup(self):
        ordering = []

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup, ordering, blowUp=True)

            def testNothing(self):
                ordering.append('test')

            def tearDown(self):
                ordering.append('tearDown')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUp', 'test', 'tearDown', 'cleanup_exc', 'tearDownModule', 'cleanup_good'])

    def test_with_errors_in_addModuleCleanup_and_setUps(self):
        ordering = []
        module_blow_up = False
        class_blow_up = False
        method_blow_up = False

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup, ordering, blowUp=True)
                if module_blow_up:
                    raise CustomError('ModuleExc')

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                ordering.append('setUpClass')
                if class_blow_up:
                    raise CustomError('ClassExc')

            def setUp(self):
                ordering.append('setUp')
                if method_blow_up:
                    raise CustomError('MethodExc')

            def testNothing(self):
                ordering.append('test')

            @classmethod
            def tearDownClass(cls):
                ordering.append('tearDownClass')
        TestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'setUp', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])
        ordering = []
        module_blow_up = True
        class_blow_up = False
        method_blow_up = False
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ModuleExc')
        self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'cleanup_exc'])
        ordering = []
        module_blow_up = False
        class_blow_up = True
        method_blow_up = False
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ClassExc')
        self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'tearDownModule', 'cleanup_exc'])
        ordering = []
        module_blow_up = False
        class_blow_up = False
        method_blow_up = True
        result = runTests(TestableTest)
        self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: MethodExc')
        self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
        self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'setUp', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])

    def test_module_cleanUp_with_multiple_classes(self):
        ordering = []

        def cleanup1():
            ordering.append('cleanup1')

        def cleanup2():
            ordering.append('cleanup2')

        def cleanup3():
            ordering.append('cleanup3')

        class Module(object):

            @staticmethod
            def setUpModule():
                ordering.append('setUpModule')
                unittest.addModuleCleanup(cleanup1)

            @staticmethod
            def tearDownModule():
                ordering.append('tearDownModule')

        class TestableTest(unittest.TestCase):

            def setUp(self):
                ordering.append('setUp')
                self.addCleanup(cleanup2)

            def testNothing(self):
                ordering.append('test')

            def tearDown(self):
                ordering.append('tearDown')

        class OtherTestableTest(unittest.TestCase):

            def setUp(self):
                ordering.append('setUp2')
                self.addCleanup(cleanup3)

            def testNothing(self):
                ordering.append('test2')

            def tearDown(self):
                ordering.append('tearDown2')
        TestableTest.__module__ = 'Module'
        OtherTestableTest.__module__ = 'Module'
        sys.modules['Module'] = Module
        runTests(TestableTest, OtherTestableTest)
        self.assertEqual(ordering, ['setUpModule', 'setUp', 'test', 'tearDown', 'cleanup2', 'setUp2', 'test2', 'tearDown2', 'cleanup3', 'tearDownModule', 'cleanup1'])

    def test_enterModuleContext(self):
        cleanups = []
        unittest.addModuleCleanup(cleanups.append, 'cleanup1')
        cm = TestCM(cleanups, 42)
        self.assertEqual(unittest.enterModuleContext(cm), 42)
        unittest.addModuleCleanup(cleanups.append, 'cleanup2')
        unittest.case.doModuleCleanups()
        self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])

    def test_enterModuleContext_arg_errors(self):

        class TestableTest(unittest.TestCase):

            def testNothing(self):
                pass
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            unittest.enterModuleContext(LacksEnterAndExit())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            unittest.enterModuleContext(LacksEnter())
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            unittest.enterModuleContext(LacksExit())
        self.assertEqual(unittest.case._module_cleanups, [])