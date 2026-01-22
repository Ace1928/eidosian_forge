import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def _classCleanUp(self, result):
    try:
        util._Janitor(self, result).postClassCleanup()
    except BaseException:
        result.addError(self, failure.Failure())