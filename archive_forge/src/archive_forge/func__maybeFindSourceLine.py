import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def _maybeFindSourceLine(testThing):
    """
    Try to find the source line of the given test thing.

    @param testThing: the test item to attempt to inspect
    @type testThing: an L{TestCase}, test method, or module, though only the
        former two have a chance to succeed
    @rtype: int
    @return: the starting source line, or -1 if one couldn't be found
    """
    method = getattr(testThing, '_testMethodName', None)
    if method is not None:
        testThing = getattr(testThing, method)
    code = getattr(testThing, '__code__', None)
    if code is not None:
        return code.co_firstlineno
    try:
        return inspect.getsourcelines(testThing)[1]
    except (OSError, TypeError):
        return -1