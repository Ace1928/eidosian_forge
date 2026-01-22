from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def _ExecuteTool(args, **extra_popen_kwargs):
    """Executes a new tool with the given args, plus the args from the cmdline.

  Args:
    args: [str], The args of the command to execute.
    **extra_popen_kwargs: [dict], kwargs to be unpacked in Popen call for tool.
  """
    execution_utils.Exec(args + sys.argv[1:], env=_GetToolEnv(), **extra_popen_kwargs)