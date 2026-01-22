from __future__ import annotations
import queue as Queue
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.interfaces import IReactorFromThreads
from twisted.python import failure
from twisted.python.threadpool import ThreadPool
def _runMultiple(tupleList):
    """
    Run a list of functions.
    """
    for f, args, kwargs in tupleList:
        f(*args, **kwargs)