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
def ExecutePythonTool(tool_dir, exec_name, *args):
    """Execute the given python script with the given args and command line.

  Args:
    tool_dir: the directory the tool is located in
    exec_name: additional path to the executable under the tool_dir
    *args: args for the command
  """
    py_path = None
    extra_popen_kwargs = {}
    if exec_name == 'gsutil':
        gsutil_py = encoding.GetEncodedValue(os.environ, 'CLOUDSDK_GSUTIL_PYTHON')
        extra_popen_kwargs['close_fds'] = False
        if gsutil_py:
            py_path = gsutil_py
    if exec_name == 'bq.py':
        bq_py = encoding.GetEncodedValue(os.environ, 'CLOUDSDK_BQ_PYTHON')
        if bq_py:
            py_path = bq_py
    _ExecuteTool(execution_utils.ArgsForPythonTool(_FullPath(tool_dir, exec_name), *args, python=py_path), **extra_popen_kwargs)