from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import functools
import multiprocessing
import sys
import threading
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.command_lib import crash_handling
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_buffer
from googlecloudsdk.command_lib.storage.tasks import task_graph as task_graph_module
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds_context_managers
from googlecloudsdk.core.util import platforms
from six.moves import queue
@crash_handling.CrashManager
def _process_factory(task_queue, task_output_queue, task_status_queue, thread_count, idle_thread_count, signal_queue, shared_process_context):
    """Create worker processes.

  This factory must run in a separate process to avoid deadlock issue,
  see go/gcloud-storage-deadlock-issue/. Although we are adding one
  extra process by doing this, it will remain idle once all the child worker
  processes are created. Thus, it does not add noticable burden on the system.

  Args:
    task_queue (multiprocessing.Queue): Holds task_graph.TaskWrapper instances.
    task_output_queue (multiprocessing.Queue): Sends information about completed
      tasks back to the main process.
    task_status_queue (multiprocessing.Queue|None): Used by task to report it
      progress to a central location.
    thread_count (int): Number of threads the process should spawn.
    idle_thread_count (multiprocessing.Semaphore): Passed on to worker threads.
    signal_queue (multiprocessing.Queue): Queue used by parent process to
      signal when a new child worker process must be created.
    shared_process_context (SharedProcessContext): Holds values from global
      state that need to be replicated in child processes.
  """
    processes = []
    while True:
        signal = signal_queue.get()
        if signal == _SHUTDOWN:
            for _ in processes:
                for _ in range(thread_count):
                    task_queue.put(_SHUTDOWN)
            break
        elif signal == _CREATE_WORKER_PROCESS:
            for _ in range(thread_count):
                idle_thread_count.release()
            process = multiprocessing_context.Process(target=_process_worker, args=(task_queue, task_output_queue, task_status_queue, thread_count, idle_thread_count, shared_process_context))
            processes.append(process)
            log.debug('Adding 1 process with {} threads. Total processes: {}. Total threads: {}.'.format(thread_count, len(processes), len(processes) * thread_count))
            process.start()
        else:
            raise errors.Error('Received invalid signal for worker process creation: {}'.format(signal))
    for process in processes:
        process.join()