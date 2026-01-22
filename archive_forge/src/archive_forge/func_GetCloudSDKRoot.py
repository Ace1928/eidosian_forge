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
def GetCloudSDKRoot():
    """Gets the directory of the root of the Cloud SDK, error if it doesn't exist.

  Raises:
    NoCloudSDKError: If there is no SDK root.

  Returns:
    str, The path to the root of the Cloud SDK.
  """
    sdk_root = config.Paths().sdk_root
    if not sdk_root:
        raise NoCloudSDKError()
    log.debug('Found Cloud SDK root: %s', sdk_root)
    return sdk_root