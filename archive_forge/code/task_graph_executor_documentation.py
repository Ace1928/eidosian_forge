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
Executes tasks from a task iterator in parallel.

    Returns:
      An integer indicating the exit code. Zero indicates no fatal errors were
        raised.
    