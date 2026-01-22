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
def CheckUpdates(command_path):
    """Check for updates and inform the user.

  Args:
    command_path: str, The '.' separated path of the command that is currently
      being run (i.e. gcloud.foo.bar).
  """
    try:
        update_manager.UpdateManager.PerformUpdateCheck(command_path=command_path)
    except Exception:
        pass