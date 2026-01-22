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
class NonThreadedExecutor(BaseExecutor):
    """A drop-in replacement non-threaded version of ThreadPoolExecutor"""

    def submit(self, fn, *args, **kwargs):
        future = NonThreadedExecutorFuture()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception:
            e, tb = sys.exc_info()[1:]
            logger.debug('Setting exception for %s to %s with traceback %s', future, e, tb)
            future.set_exception_info(e, tb)
        return future

    def shutdown(self, wait=True):
        pass