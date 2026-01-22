import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _ebDeferTestMethod(self, f, result):
    todo = self.getTodo()
    if todo is not None and todo.expected(f):
        result.addExpectedFailure(self, f, todo)
    elif f.check(self.failureException, FailTest):
        result.addFailure(self, f)
    elif f.check(KeyboardInterrupt):
        result.addError(self, f)
        result.stop()
    elif f.check(SkipTest):
        result.addSkip(self, self._getSkipReason(getattr(self, self._testMethodName), f.value))
    else:
        result.addError(self, f)