import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class TestLoader(unittest.TestLoader):
    """Custom TestLoader to extend the stock python one."""
    suiteClass = TestSuite
    test_func_names: Dict[str, List[str]] = {}

    def loadTestsFromModuleNames(self, names):
        """use a custom means to load tests from modules.

        There is an undesirable glitch in the python TestLoader where a
        import error is ignore. We think this can be solved by ensuring the
        requested name is resolvable, if its not raising the original error.
        """
        result = self.suiteClass()
        for name in names:
            result.addTests(self.loadTestsFromModuleName(name))
        return result

    def loadTestsFromModuleName(self, name):
        result = self.suiteClass()
        module = pyutils.get_named_object(name)
        result.addTests(self.loadTestsFromModule(module))
        return result

    def getTestCaseNames(self, test_case_class):
        test_fn_names = self.test_func_names.get(test_case_class, None)
        if test_fn_names is not None:
            return test_fn_names
        test_fn_names = unittest.TestLoader.getTestCaseNames(self, test_case_class)
        self.test_func_names[test_case_class] = test_fn_names
        return test_fn_names