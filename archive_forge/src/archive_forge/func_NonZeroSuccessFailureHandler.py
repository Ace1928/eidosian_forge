from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def NonZeroSuccessFailureHandler(result_holder, show_exec_error=False):
    """Processing for subprocess where non-zero exit status is not always failure.

  Uses rule of thumb that defines success as:
  - a process with zero exit status OR
  - a process with non-zero exit status AND some stdout output.

  All others are considered failed.

  Args:
    result_holder: OperationResult, result of command execution
    show_exec_error: bool, if true log the process command and exit status the
      terminal for failed executions.

  Returns:
    None. Sets the failed attribute of the result_holder.
  """
    if result_holder.exit_code != 0 and (not result_holder.stdout):
        result_holder.failed = True
    if show_exec_error and result_holder.failed:
        _LogDefaultOperationFailure(result_holder)