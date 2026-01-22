from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def WaitForBatchCompletion(ssh_threads, exit_statuses):
    """Waits for all the running ssh threads to complete.

  Exits with a nonzero code, if there are any non-zero exit status in ssh
  command execution. This ensures that if any command failed on a worker,
  we don't end up returning 0 for a value.

  Args:
    ssh_threads: List of ssh threads.
    exit_statuses: List of exit status of each ssh execution.
  """
    for ssh_thread in ssh_threads:
        ssh_thread.join()
    for status in exit_statuses:
        if status:
            sys.exit(status)