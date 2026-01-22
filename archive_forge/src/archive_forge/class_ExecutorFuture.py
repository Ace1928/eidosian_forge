from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
class ExecutorFuture(object):

    def __init__(self, future):
        """A future returned from the executor

        Currently, it is just a wrapper around a concurrent.futures.Future.
        However, this can eventually grow to implement the needed functionality
        of concurrent.futures.Future if we move off of the library and not
        affect the rest of the codebase.

        :type future: concurrent.futures.Future
        :param future: The underlying future
        """
        self._future = future

    def result(self):
        return self._future.result()

    def add_done_callback(self, fn):
        """Adds a callback to be completed once future is done

        :parm fn: A callable that takes no arguments. Note that is different
            than concurrent.futures.Future.add_done_callback that requires
            a single argument for the future.
        """

        def done_callback(future_passed_to_callback):
            return fn()
        self._future.add_done_callback(done_callback)

    def done(self):
        return self._future.done()