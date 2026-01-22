from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import codecs
from collections import namedtuple
import copy
import getopt
import json
import logging
import os
import signal
import sys
import textwrap
import threading
import time
import traceback
import boto
from boto.storage_uri import StorageUri
import gslib
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import ServiceException
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.cs_api_map import ApiSelector
from gslib.cs_api_map import GsutilApiMapFactory
from gslib.exception import CommandException
from gslib.help_provider import HelpProvider
from gslib.metrics import CaptureThreadStatException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.name_expansion import CopyObjectInfo
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import NameExpansionResult
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadThread
from gslib.sig_handling import ChildProcessSignalHandler
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import KillProcess
from gslib.sig_handling import MultithreadedMainSignalHandler
from gslib.sig_handling import RegisterSignalHandler
from gslib.storage_url import HaveFileUrls
from gslib.storage_url import HaveProviderUrls
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import PerformanceSummaryMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.ui_controller import MainThreadUIQueue
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import GetMaxConcurrentCompressedUploads
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
import gslib.utils.parallelism_framework_util
from gslib.utils.parallelism_framework_util import AtomicDict
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.parallelism_framework_util import ProcessAndThreadSafeInt
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.parallelism_framework_util import SEEK_AHEAD_JOIN_TIMEOUT
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from gslib.utils.parallelism_framework_util import UI_THREAD_JOIN_TIMEOUT
from gslib.utils.parallelism_framework_util import ZERO_TASKS_TO_DO_ARGUMENT
from gslib.utils.rsync_util import RsyncDiffToApply
from gslib.utils.shim_util import GcloudStorageCommandMixin
from gslib.utils.system_util import GetTermLines
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import GetNonMetadataHeaders
from gslib.utils.translation_helper import PRIVATE_DEFAULT_OBJ_ACL
from gslib.wildcard_iterator import CreateWildcardIterator
from six.moves import queue as Queue
def _ParallelApply(self, func, args_iterator, exception_handler, caller_id, arg_checker, process_count, thread_count, should_return_results, fail_on_error, seek_ahead_iterator=None, parallel_operations_override=None):
    """Dispatches input arguments across a thread/process pool.

    Pools are composed of parallel OS processes and/or Python threads,
    based on options (-m or not) and settings in the user's config file.

    If only one OS process is requested/available, dispatch requests across
    threads in the current OS process.

    In the multi-process case, we will create one pool of worker processes for
    each level of the tree of recursive calls to Apply. E.g., if A calls
    Apply(B), and B ultimately calls Apply(C) followed by Apply(D), then we
    will only create two sets of worker processes - B will execute in the first,
    and C and D will execute in the second. If C is then changed to call
    Apply(E) and D is changed to call Apply(F), then we will automatically
    create a third set of processes (lazily, when needed) that will be used to
    execute calls to E and F. This might look something like:

    Pool1 Executes:                B
                                  / \\
    Pool2 Executes:              C   D
                                /     \\
    Pool3 Executes:            E       F

    Apply's parallelism is generally broken up into 4 cases:
    - If process_count == thread_count == 1, then all tasks will be executed
      by _SequentialApply.
    - If process_count > 1 and thread_count == 1, then the main thread will
      create a new pool of processes (if they don't already exist) and each of
      those processes will execute the tasks in a single thread.
    - If process_count == 1 and thread_count > 1, then this process will create
      a new pool of threads to execute the tasks.
    - If process_count > 1 and thread_count > 1, then the main thread will
      create a new pool of processes (if they don't already exist) and each of
      those processes will, upon creation, create a pool of threads to
      execute the tasks.

    Args:
      caller_id: The caller ID unique to this call to command.Apply.
      See command.Apply for description of other arguments.
    """
    global glob_status_queue, ui_controller
    is_main_thread = self.recursive_apply_level == 0
    if parallel_operations_override == self.ParallelOverrideReason.SLICE and self.recursive_apply_level <= 1:
        glob_status_queue.put(PerformanceSummaryMessage(time.time(), True))
    if not IS_WINDOWS and is_main_thread:
        for signal_num in (signal.SIGINT, signal.SIGTERM):
            RegisterSignalHandler(signal_num, MultithreadedMainSignalHandler, is_final_handler=True)
    if not task_queues:
        if process_count > 1:
            task_queues.append(_NewMultiprocessingQueue())
        else:
            task_queue = _NewThreadsafeQueue()
            task_queues.append(task_queue)
            WorkerPool(thread_count, self.logger, task_queue=task_queue, bucket_storage_uri_class=self.bucket_storage_uri_class, gsutil_api_map=self.gsutil_api_map, debug=self.debug, status_queue=glob_status_queue, headers=self.non_metadata_headers, perf_trace_token=self.perf_trace_token, trace_token=self.trace_token, user_project=self.user_project)
    if process_count > 1:
        try:
            if not is_main_thread:
                worker_checking_level_lock.acquire()
            if self.recursive_apply_level >= current_max_recursive_level.GetValue():
                with need_pool_or_done_cond:
                    if is_main_thread:
                        self._CreateNewConsumerPool(process_count, thread_count, glob_status_queue)
                    else:
                        new_pool_needed.Reset(reset_value=1)
                        need_pool_or_done_cond.notify_all()
                        need_pool_or_done_cond.wait()
        finally:
            if not is_main_thread:
                worker_checking_level_lock.release()
    elif not is_main_thread:
        try:
            worker_checking_level_lock.acquire()
            if self.recursive_apply_level > _GetCurrentMaxRecursiveLevel():
                _IncrementCurrentMaxRecursiveLevel()
                task_queue = _NewThreadsafeQueue()
                task_queues.append(task_queue)
                WorkerPool(thread_count, self.logger, task_queue=task_queue, bucket_storage_uri_class=self.bucket_storage_uri_class, gsutil_api_map=self.gsutil_api_map, debug=self.debug, status_queue=glob_status_queue, headers=self.non_metadata_headers, perf_trace_token=self.perf_trace_token, trace_token=self.trace_token, user_project=self.user_project)
        finally:
            worker_checking_level_lock.release()
    task_queue = task_queues[self.recursive_apply_level]
    if seek_ahead_iterator and (not is_main_thread):
        seek_ahead_iterator = None
    producer_thread = ProducerThread(copy.copy(self), args_iterator, caller_id, func, task_queue, should_return_results, exception_handler, arg_checker, fail_on_error, seek_ahead_iterator=seek_ahead_iterator, status_queue=glob_status_queue if is_main_thread else None)
    ui_thread = None
    if is_main_thread:
        ui_thread = UIThread(glob_status_queue, sys.stderr, ui_controller)
    while True:
        with need_pool_or_done_cond:
            if call_completed_map[caller_id]:
                break
            elif process_count > 1 and is_main_thread and new_pool_needed.GetValue():
                new_pool_needed.Reset()
                self._CreateNewConsumerPool(process_count, thread_count, glob_status_queue)
                need_pool_or_done_cond.notify_all()
            need_pool_or_done_cond.wait()
    if is_main_thread:
        PutToQueueWithTimeout(glob_status_queue, ZERO_TASKS_TO_DO_ARGUMENT)
        ui_thread.join(timeout=UI_THREAD_JOIN_TIMEOUT)
        self._ProcessSourceUrlTypes(producer_thread.args_iterator)
    if producer_thread.unknown_exception:
        raise producer_thread.unknown_exception
    if producer_thread.iterator_exception and fail_on_error:
        raise producer_thread.iterator_exception
    if is_main_thread and (not parallel_operations_override):
        PutToQueueWithTimeout(glob_status_queue, FinalMessage(time.time()))