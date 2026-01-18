from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Wrapper for execution_utils.Subprocess that optionally captures logs.

  Args:
    args: [str], The arguments to execute.  The first argument is the command.
    capture_logs_fn: str, If set, save logs to the specified filename.

  Returns:
    subprocess.Popen or execution_utils.SubprocessTimeoutWrapper, The running
      subprocess.
  