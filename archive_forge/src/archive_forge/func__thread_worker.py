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
def _thread_worker(task_queue, task_output_queue, task_status_queue, idle_thread_count):
    """A consumer thread run in a child process.

  Args:
    task_queue (multiprocessing.Queue): Holds task_graph.TaskWrapper instances.
    task_output_queue (multiprocessing.Queue): Sends information about completed
      tasks back to the main process.
    task_status_queue (multiprocessing.Queue|None): Used by task to report it
      progress to a central location.
    idle_thread_count (multiprocessing.Semaphore): Keeps track of how many
      threads are busy. Useful for spawning new workers if all threads are busy.
  """
    while True:
        with _task_queue_lock():
            task_wrapper = task_queue.get()
        if task_wrapper == _SHUTDOWN:
            break
        idle_thread_count.acquire()
        task_execution_error = None
        try:
            task_output = task_wrapper.task.execute(task_status_queue=task_status_queue)
        except Exception as exception:
            task_execution_error = exception
            log.error(exception)
            log.debug(exception, exc_info=sys.exc_info())
            if isinstance(exception, errors.FatalError):
                task_output = task.Output(additional_task_iterators=None, messages=[task.Message(topic=task.Topic.FATAL_ERROR, payload={})])
            elif task_wrapper.task.change_exit_code:
                task_output = task.Output(additional_task_iterators=None, messages=[task.Message(topic=task.Topic.CHANGE_EXIT_CODE, payload={})])
            else:
                task_output = None
        finally:
            task_wrapper.task.exit_handler(task_execution_error, task_status_queue)
        task_output_queue.put((task_wrapper, task_output))
        idle_thread_count.release()