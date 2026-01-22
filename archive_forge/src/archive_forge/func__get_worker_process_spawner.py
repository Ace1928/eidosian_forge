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
@contextlib.contextmanager
def _get_worker_process_spawner(self, shared_process_context):
    """Creates a worker process spawner.

    Must be used as a context manager since the worker process spawner must be
    non-daemonic in order to start child processes, but non-daemonic child
    processes block parent processes from exiting, so if there are any failures
    after the worker process spawner is started, gcloud storage will fail to
    exit, unless we put the shutdown logic in a `finally` block.

    Args:
      shared_process_context (SharedProcessContext): Holds values from global
        state that need to be replicated in child processes.

    Yields:
      None, allows body of a `with` statement to execute.
    """
    worker_process_spawner = multiprocessing_context.Process(target=_process_factory, args=(self._task_queue, self._task_output_queue, self._task_status_queue, self._thread_count, self._idle_thread_count, self._signal_queue, shared_process_context))
    try:
        worker_process_spawner.start()
        yield
    finally:
        self._signal_queue.put(_SHUTDOWN)
        worker_process_spawner.join()