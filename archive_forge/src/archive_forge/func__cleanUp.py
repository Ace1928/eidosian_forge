import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _cleanUp(self, result):
    try:
        clean = util._Janitor(self, result).postCaseCleanup()
        if not clean:
            self._passed = False
    except BaseException:
        result.addError(self, failure.Failure())
        self._passed = False
    for error in self._observer.getErrors():
        result.addError(self, error)
        self._passed = False
    self.flushLoggedErrors()
    self._removeObserver()
    if self._passed:
        result.addSuccess(self)