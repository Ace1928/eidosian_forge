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
def ExecuteCMDTool(tool_dir, exec_name, *args):
    """Execute the given batch file with the given args.

  Args:
    tool_dir: the directory the tool is located in
    exec_name: additional path to the executable under the tool_dir
    *args: args for the command
  """
    _ExecuteTool(execution_utils.ArgsForCMDTool(_FullPath(tool_dir, exec_name), *args))