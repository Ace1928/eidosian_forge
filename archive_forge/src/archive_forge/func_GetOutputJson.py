from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os.path
import subprocess
import threading
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def GetOutputJson(cmd, timeout_sec, show_stderr=True):
    """Run command and get its JSON stdout as a parsed dict.

  Args:
    cmd: List of executable and arg strings.
    timeout_sec: Command will be killed if it exceeds this.
    show_stderr: False to suppress stderr from the command.

  Returns:
    Parsed JSON.
  """
    stdout = _GetStdout(cmd, timeout_sec, show_stderr=show_stderr)
    return json.loads(stdout.strip())