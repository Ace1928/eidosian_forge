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
@_store_exception
def _handle_task_output(self):
    """Updates a dependency graph based on information from executed tasks."""
    while True:
        output = self._task_output_queue.get()
        if output == _SHUTDOWN:
            break
        executed_task_wrapper, task_output = output
        if task_output and task_output.messages:
            for message in task_output.messages:
                if message.topic in (task.Topic.CHANGE_EXIT_CODE, task.Topic.FATAL_ERROR):
                    self._exit_code = 1
                    if message.topic == task.Topic.FATAL_ERROR:
                        self._accepting_new_tasks = False
        submittable_tasks = self._task_graph.update_from_executed_task(executed_task_wrapper, task_output)
        for task_wrapper in submittable_tasks:
            task_wrapper.is_submitted = True
            self._executable_tasks.put(task_wrapper)