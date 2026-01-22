from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
class BatchCompletionCallBack(object):
    """Callback to keep track of completed results and schedule the next tasks.

    This callable is executed by the parent process whenever a worker process
    has completed a batch of tasks.

    It is used for progress reporting, to update estimate of the batch
    processing duration and to schedule the next batch of tasks to be
    processed.

    It is assumed that this callback will always be triggered by the backend
    right after the end of a task, in case of success as well as in case of
    failure.
    """

    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel
        self.parallel_call_id = parallel._call_id
        self.job = None
        if not parallel._backend.supports_retrieve_callback:
            self.status = None
        else:
            self.status = TASK_PENDING

    def register_job(self, job):
        """Register the object returned by `apply_async`."""
        self.job = job

    def get_result(self, timeout):
        """Returns the raw result of the task that was submitted.

        If the task raised an exception rather than returning, this same
        exception will be raised instead.

        If the backend supports the retrieval callback, it is assumed that this
        method is only called after the result has been registered. It is
        ensured by checking that `self.status(timeout)` does not return
        TASK_PENDING. In this case, `get_result` directly returns the
        registered result (or raise the registered exception).

        For other backends, there are no such assumptions, but `get_result`
        still needs to synchronously retrieve the result before it can
        return it or raise. It will block at most `self.timeout` seconds
        waiting for retrieval to complete, after that it raises a TimeoutError.
        """
        backend = self.parallel._backend
        if backend.supports_retrieve_callback:
            return self._return_or_raise()
        try:
            if backend.supports_timeout:
                result = self.job.get(timeout=timeout)
            else:
                result = self.job.get()
            outcome = dict(result=result, status=TASK_DONE)
        except BaseException as e:
            outcome = dict(result=e, status=TASK_ERROR)
        self._register_outcome(outcome)
        return self._return_or_raise()

    def _return_or_raise(self):
        try:
            if self.status == TASK_ERROR:
                raise self._result
            return self._result
        finally:
            del self._result

    def get_status(self, timeout):
        """Get the status of the task.

        This function also checks if the timeout has been reached and register
        the TimeoutError outcome when it is the case.
        """
        if timeout is None or self.status != TASK_PENDING:
            return self.status
        now = time.time()
        if not hasattr(self, '_completion_timeout_counter'):
            self._completion_timeout_counter = now
        if now - self._completion_timeout_counter > timeout:
            outcome = dict(result=TimeoutError(), status=TASK_ERROR)
            self._register_outcome(outcome)
        return self.status

    def __call__(self, out):
        """Function called by the callback thread after a job is completed."""
        if not self.parallel._backend.supports_retrieve_callback:
            self._dispatch_new()
            return
        with self.parallel._lock:
            if self.parallel._call_id != self.parallel_call_id:
                return
            if self.parallel._aborting:
                return
            job_succeeded = self._retrieve_result(out)
        if job_succeeded:
            self._dispatch_new()

    def _dispatch_new(self):
        """Schedule the next batch of tasks to be processed."""
        this_batch_duration = time.time() - self.dispatch_timestamp
        self.parallel._backend.batch_completed(self.batch_size, this_batch_duration)
        with self.parallel._lock:
            self.parallel.n_completed_tasks += self.batch_size
            self.parallel.print_progress()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    def _retrieve_result(self, out):
        """Fetch and register the outcome of a task.

        Return True if the task succeeded, False otherwise.
        This function is only called by backends that support retrieving
        the task result in the callback thread.
        """
        try:
            result = self.parallel._backend.retrieve_result_callback(out)
            outcome = dict(status=TASK_DONE, result=result)
        except BaseException as e:
            e.__traceback__ = None
            outcome = dict(result=e, status=TASK_ERROR)
        self._register_outcome(outcome)
        return outcome['status'] != TASK_ERROR

    def _register_outcome(self, outcome):
        """Register the outcome of a task.

        This method can be called only once, future calls will be ignored.
        """
        with self.parallel._lock:
            if self.status not in (TASK_PENDING, None):
                return
            self.status = outcome['status']
        self._result = outcome['result']
        self.job = None
        if self.status == TASK_ERROR:
            self.parallel._exception = True
            self.parallel._aborting = True