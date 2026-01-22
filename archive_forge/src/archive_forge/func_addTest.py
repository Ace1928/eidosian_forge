import sys
from . import case
from . import util
def addTest(self, test):
    if not callable(test):
        raise TypeError('{} is not callable'.format(repr(test)))
    if isinstance(test, type) and issubclass(test, (case.TestCase, TestSuite)):
        raise TypeError('TestCases and TestSuites must be instantiated before passing them to addTest()')
    self._tests.append(test)