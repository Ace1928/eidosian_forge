from __future__ import annotations
import os
import sys
import unittest as pyunit
from hashlib import md5
from operator import attrgetter
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator
from hamcrest import assert_that, equal_to, has_properties
from hamcrest.core.matcher import Matcher
from twisted.python import filepath, util
from twisted.python.modules import PythonAttribute, PythonModule, getModule
from twisted.python.reflect import ModuleNotFound
from twisted.trial import reporter, runner, unittest
from twisted.trial._asyncrunner import _iterateTests
from twisted.trial.itrial import ITestCase
from twisted.trial.test import packages
from .matchers import after
class LoaderTests(packages.SysPathManglingTest):
    """
    Tests for L{trial.TestLoader}.
    """

    def setUp(self) -> None:
        self.loader = runner.TestLoader()
        packages.SysPathManglingTest.setUp(self)

    def test_sortCases(self) -> None:
        from twisted.trial.test import sample
        suite = self.loader.loadClass(sample.AlphabetTest)
        self.assertEqual(['test_a', 'test_b', 'test_c'], [test._testMethodName for test in suite._tests])
        newOrder = ['test_b', 'test_c', 'test_a']
        sortDict = dict(zip(newOrder, range(3)))
        self.loader.sorter = lambda x: sortDict.get(x.shortDescription(), -1)
        suite = self.loader.loadClass(sample.AlphabetTest)
        self.assertEqual(newOrder, [test._testMethodName for test in suite._tests])

    def test_loadFailure(self) -> None:
        """
        Loading a test that fails and getting the result of it ends up with one
        test ran and one failure.
        """
        suite = self.loader.loadByName('twisted.trial.test.erroneous.TestRegularFail.test_fail')
        result = reporter.TestResult()
        suite.run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures), 1)

    def test_loadBadDecorator(self) -> None:
        """
        A decorated test method for which the decorator has failed to set the
        method's __name__ correctly is loaded and its name in the class scope
        discovered.
        """
        from twisted.trial.test import sample
        suite = self.loader.loadAnything(sample.DecorationTest.test_badDecorator, parent=sample.DecorationTest, qualName=['sample', 'DecorationTest', 'test_badDecorator'])
        self.assertEqual(1, suite.countTestCases())
        self.assertEqual('test_badDecorator', suite._testMethodName)

    def test_loadGoodDecorator(self) -> None:
        """
        A decorated test method for which the decorator has set the method's
        __name__ correctly is loaded and the only name by which it goes is used.
        """
        from twisted.trial.test import sample
        suite = self.loader.loadAnything(sample.DecorationTest.test_goodDecorator, parent=sample.DecorationTest, qualName=['sample', 'DecorationTest', 'test_goodDecorator'])
        self.assertEqual(1, suite.countTestCases())
        self.assertEqual('test_goodDecorator', suite._testMethodName)

    def test_loadRenamedDecorator(self) -> None:
        """
        Load a decorated method which has been copied to a new name inside the
        class.  Thus its __name__ and its key in the class's __dict__ no
        longer match.
        """
        from twisted.trial.test import sample
        suite = self.loader.loadAnything(sample.DecorationTest.test_renamedDecorator, parent=sample.DecorationTest, qualName=['sample', 'DecorationTest', 'test_renamedDecorator'])
        self.assertEqual(1, suite.countTestCases())
        self.assertEqual('test_renamedDecorator', suite._testMethodName)

    def test_loadClass(self) -> None:
        from twisted.trial.test import sample
        suite = self.loader.loadClass(sample.FooTest)
        self.assertEqual(2, suite.countTestCases())
        self.assertEqual(['test_bar', 'test_foo'], [test._testMethodName for test in suite._tests])

    def test_loadNonClass(self) -> None:
        from twisted.trial.test import sample
        self.assertRaises(TypeError, self.loader.loadClass, sample)
        self.assertRaises(TypeError, self.loader.loadClass, sample.FooTest.test_foo)
        self.assertRaises(TypeError, self.loader.loadClass, 'string')
        self.assertRaises(TypeError, self.loader.loadClass, ('foo', 'bar'))

    def test_loadNonTestCase(self) -> None:
        from twisted.trial.test import sample
        self.assertRaises(ValueError, self.loader.loadClass, sample.NotATest)

    def test_loadModule(self) -> None:
        from twisted.trial.test import sample
        suite = self.loader.loadModule(sample)
        self.assertEqual(10, suite.countTestCases())

    def test_loadNonModule(self) -> None:
        from twisted.trial.test import sample
        self.assertRaises(TypeError, self.loader.loadModule, sample.FooTest)
        self.assertRaises(TypeError, self.loader.loadModule, sample.FooTest.test_foo)
        self.assertRaises(TypeError, self.loader.loadModule, 'string')
        self.assertRaises(TypeError, self.loader.loadModule, ('foo', 'bar'))

    def test_loadPackage(self) -> None:
        import goodpackage
        suite = self.loader.loadPackage(goodpackage)
        self.assertEqual(7, suite.countTestCases())

    def test_loadNonPackage(self) -> None:
        from twisted.trial.test import sample
        self.assertRaises(TypeError, self.loader.loadPackage, sample.FooTest)
        self.assertRaises(TypeError, self.loader.loadPackage, sample.FooTest.test_foo)
        self.assertRaises(TypeError, self.loader.loadPackage, 'string')
        self.assertRaises(TypeError, self.loader.loadPackage, ('foo', 'bar'))

    def test_loadModuleAsPackage(self) -> None:
        from twisted.trial.test import sample
        self.assertRaises(TypeError, self.loader.loadPackage, sample)

    def test_loadPackageRecursive(self) -> None:
        import goodpackage
        suite = self.loader.loadPackage(goodpackage, recurse=True)
        self.assertEqual(14, suite.countTestCases())

    def test_loadAnythingOnModule(self) -> None:
        from twisted.trial.test import sample
        suite = self.loader.loadAnything(sample)
        self.assertEqual(sample.__name__, suite._tests[0]._tests[0].__class__.__module__)

    def test_loadAnythingOnClass(self) -> None:
        from twisted.trial.test import sample
        suite = self.loader.loadAnything(sample.FooTest)
        self.assertEqual(2, suite.countTestCases())

    def test_loadAnythingOnPackage(self) -> None:
        import goodpackage
        suite = self.loader.loadAnything(goodpackage)
        self.assertTrue(isinstance(suite, self.loader.suiteFactory))
        self.assertEqual(7, suite.countTestCases())

    def test_loadAnythingOnPackageRecursive(self) -> None:
        import goodpackage
        suite = self.loader.loadAnything(goodpackage, recurse=True)
        self.assertTrue(isinstance(suite, self.loader.suiteFactory))
        self.assertEqual(14, suite.countTestCases())

    def test_loadAnythingOnString(self) -> None:
        self.assertRaises(TypeError, self.loader.loadAnything, 'goodpackage')

    def test_importErrors(self) -> None:
        import package
        suite = self.loader.loadPackage(package, recurse=True)
        result = reporter.Reporter()
        suite.run(result)
        self.assertEqual(False, result.wasSuccessful())
        self.assertEqual(2, len(result.errors))
        errors = [test.id() for test, error in result.errors]
        errors.sort()
        self.assertEqual(errors, ['package.test_bad_module', 'package.test_import_module'])

    def test_differentInstances(self) -> None:
        """
        L{TestLoader.loadClass} returns a suite with each test method
        represented by a different instances of the L{TestCase} they are
        defined on.
        """

        class DistinctInstances(pyunit.TestCase):

            def test_1(self) -> None:
                self.first = 'test1Run'

            def test_2(self) -> None:
                self.assertFalse(hasattr(self, 'first'))
        suite = self.loader.loadClass(DistinctInstances)
        result = reporter.Reporter()
        suite.run(result)
        self.assertTrue(result.wasSuccessful())

    def test_loadModuleWith_test_suite(self) -> None:
        """
        Check that C{test_suite} is used when present and other L{TestCase}s are
        not included.
        """
        from twisted.trial.test import mockcustomsuite
        suite = self.loader.loadModule(mockcustomsuite)
        self.assertEqual(0, suite.countTestCases())
        self.assertEqual('MyCustomSuite', getattr(suite, 'name', None))

    def test_loadModuleWith_testSuite(self) -> None:
        """
        Check that C{testSuite} is used when present and other L{TestCase}s are
        not included.
        """
        from twisted.trial.test import mockcustomsuite2
        suite = self.loader.loadModule(mockcustomsuite2)
        self.assertEqual(0, suite.countTestCases())
        self.assertEqual('MyCustomSuite', getattr(suite, 'name', None))

    def test_loadModuleWithBothCustom(self) -> None:
        """
        Check that if C{testSuite} and C{test_suite} are both present in a
        module then C{testSuite} gets priority.
        """
        from twisted.trial.test import mockcustomsuite3
        suite = self.loader.loadModule(mockcustomsuite3)
        self.assertEqual('testSuite', getattr(suite, 'name', None))

    def test_customLoadRaisesAttributeError(self) -> None:
        """
        Make sure that any C{AttributeError}s raised by C{testSuite} are not
        swallowed by L{TestLoader}.
        """

        def testSuite() -> None:
            raise AttributeError('should be reraised')
        from twisted.trial.test import mockcustomsuite2
        mockcustomsuite2.testSuite, original = (testSuite, mockcustomsuite2.testSuite)
        try:
            self.assertRaises(AttributeError, self.loader.loadModule, mockcustomsuite2)
        finally:
            mockcustomsuite2.testSuite = original

    def assertSuitesEqual(self, test1: pyunit.TestCase | pyunit.TestSuite, test2: pyunit.TestCase | pyunit.TestSuite) -> None:
        names1 = testNames(test1)
        names2 = testNames(test2)
        names1.sort()
        names2.sort()
        self.assertEqual(names1, names2)

    def test_loadByNamesDuplicate(self) -> None:
        """
        Check that loadByNames ignores duplicate names
        """
        module = 'twisted.trial.test.test_log'
        suite1 = self.loader.loadByNames([module, module], True)
        suite2 = self.loader.loadByName(module, True)
        self.assertSuitesEqual(suite1, suite2)

    def test_loadByNamesPreservesOrder(self) -> None:
        """
        L{TestLoader.loadByNames} preserves the order of tests provided to it.
        """
        modules = ['inheritancepackage.test_x.A.test_foo', 'twisted.trial.test.sample', 'goodpackage', 'twisted.trial.test.test_log', 'twisted.trial.test.sample.FooTest', 'package.test_module']
        suite1 = self.loader.loadByNames(modules)
        suite2 = runner.TestSuite(map(self.loader.loadByName, modules))
        self.assertEqual(testNames(suite1), testNames(suite2))

    def test_loadDifferentNames(self) -> None:
        """
        Check that loadByNames loads all the names that it is given
        """
        modules = ['goodpackage', 'package.test_module']
        suite1 = self.loader.loadByNames(modules)
        suite2 = runner.TestSuite(map(self.loader.loadByName, modules))
        self.assertSuitesEqual(suite1, suite2)

    def test_loadInheritedMethods(self) -> None:
        """
        Check that test methods names which are inherited from are all
        loaded rather than just one.
        """
        methods = ['inheritancepackage.test_x.A.test_foo', 'inheritancepackage.test_x.B.test_foo']
        suite1 = self.loader.loadByNames(methods)
        suite2 = runner.TestSuite(map(self.loader.loadByName, methods))
        self.assertSuitesEqual(suite1, suite2)