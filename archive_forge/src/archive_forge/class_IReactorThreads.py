from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReactorThreads(IReactorFromThreads, IReactorInThreads):
    """
    Dispatch methods to be run in threads.

    Internally, this should use a thread pool and dispatch methods to them.
    """

    def getThreadPool() -> 'ThreadPool':
        """
        Return the threadpool used by L{IReactorInThreads.callInThread}.
        Create it first if necessary.
        """

    def suggestThreadPoolSize(size: int) -> None:
        """
        Suggest the size of the internal threadpool used to dispatch functions
        passed to L{IReactorInThreads.callInThread}.
        """