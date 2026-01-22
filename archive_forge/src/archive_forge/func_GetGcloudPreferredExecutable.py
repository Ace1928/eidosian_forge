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
def GetGcloudPreferredExecutable(exe):
    """Finds the path to an executable, preferring the gcloud packaged version.

  Args:
    exe: Name of the executable.

  Returns:
    Path to the executable.
  Raises:
    EnvironmentError: The executable can't be found.
  """
    path = _FindOrInstallComponent(exe) or file_utils.FindExecutableOnPath(exe)
    if not path:
        raise EnvironmentError('Unable to locate %s.' % exe)
    return path