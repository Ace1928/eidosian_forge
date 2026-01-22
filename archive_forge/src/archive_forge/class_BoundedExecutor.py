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
class BoundedExecutor(object):
    EXECUTOR_CLS = futures.ThreadPoolExecutor

    def __init__(self, max_size, max_num_threads, tag_semaphores=None, executor_cls=None):
        """An executor implentation that has a maximum queued up tasks

        The executor will block if the number of tasks that have been
        submitted and is currently working on is past its maximum.

        :params max_size: The maximum number of inflight futures. An inflight
            future means that the task is either queued up or is currently
            being executed. A size of None or 0 means that the executor will
            have no bound in terms of the number of inflight futures.

        :params max_num_threads: The maximum number of threads the executor
            uses.

        :type tag_semaphores: dict
        :params tag_semaphores: A dictionary where the key is the name of the
            tag and the value is the semaphore to use when limiting the
            number of tasks the executor is processing at a time.

        :type executor_cls: BaseExecutor
        :param underlying_executor_cls: The executor class that
            get bounded by this executor. If None is provided, the
            concurrent.futures.ThreadPoolExecutor class is used.
        """
        self._max_num_threads = max_num_threads
        if executor_cls is None:
            executor_cls = self.EXECUTOR_CLS
        self._executor = executor_cls(max_workers=self._max_num_threads)
        self._semaphore = TaskSemaphore(max_size)
        self._tag_semaphores = tag_semaphores

    def submit(self, task, tag=None, block=True):
        """Submit a task to complete

        :type task: s3transfer.tasks.Task
        :param task: The task to run __call__ on


        :type tag: s3transfer.futures.TaskTag
        :param tag: An optional tag to associate to the task. This
            is used to override which semaphore to use.

        :type block: boolean
        :param block: True if to wait till it is possible to submit a task.
            False, if not to wait and raise an error if not able to submit
            a task.

        :returns: The future assocaited to the submitted task
        """
        semaphore = self._semaphore
        if tag:
            semaphore = self._tag_semaphores[tag]
        acquire_token = semaphore.acquire(task.transfer_id, block)
        release_callback = FunctionContainer(semaphore.release, task.transfer_id, acquire_token)
        future = ExecutorFuture(self._executor.submit(task))
        future.add_done_callback(release_callback)
        return future

    def shutdown(self, wait=True):
        self._executor.shutdown(wait)