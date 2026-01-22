import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
class testcase(cls, SynchronousTestCase):
    __module__ = cls.__module__
    if reactor in cls.skippedReactors:
        skip = cls.skippedReactors[reactor]
    try:
        reactorFactory = namedAny(reactor)
    except BaseException:
        skip = Failure().getErrorMessage()