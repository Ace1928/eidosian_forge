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
def GetActiveProjectAndAccount():
    """Get the active project name and account for the active credentials.

  For use with wrapping legacy tools that take projects and credentials on
  the command line.

  Returns:
    (str, str), A tuple whose first element is the project, and whose second
    element is the account.
  """
    project_name = properties.VALUES.core.project.Get(validate=False)
    account = properties.VALUES.core.account.Get(validate=False)
    return (project_name, account)