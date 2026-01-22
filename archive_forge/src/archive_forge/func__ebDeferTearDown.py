import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _ebDeferTearDown(self, failure, result):
    result.addError(self, failure)
    if failure.check(KeyboardInterrupt):
        result.stop()
    self._passed = False