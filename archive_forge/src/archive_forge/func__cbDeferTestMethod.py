import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _cbDeferTestMethod(self, ignored, result):
    if self.getTodo() is not None:
        result.addUnexpectedSuccess(self, self.getTodo())
    else:
        self._passed = True
    return ignored