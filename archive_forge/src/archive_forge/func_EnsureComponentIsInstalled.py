from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def EnsureComponentIsInstalled(component_id, for_text):
    """Ensures that the specified component is installed.

  Args:
    component_id: str, the name of the component
    for_text: str, the text explaining what the component is necessary for

  Raises:
    NoCloudSDKError: If a Cloud SDK installation is not found.
  """
    msg = 'You need the [{0}] component to use the {1}.'.format(component_id, for_text)
    try:
        update_manager.UpdateManager.EnsureInstalledAndRestart([component_id], msg=msg)
    except local_state.InvalidSDKRootError:
        raise NoCloudSDKError()