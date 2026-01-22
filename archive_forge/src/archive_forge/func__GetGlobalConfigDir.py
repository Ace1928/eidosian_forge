from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
def _GetGlobalConfigDir():
    """Returns the path to the user's global config area.

  Returns:
    str: The path to the user's global config area.
  """
    global_config_dir = encoding.GetEncodedValue(os.environ, CLOUDSDK_CONFIG)
    if global_config_dir:
        return global_config_dir
    if platforms.OperatingSystem.Current() != platforms.OperatingSystem.WINDOWS:
        return os.path.join(file_utils.GetHomeDir(), '.config', _CLOUDSDK_GLOBAL_CONFIG_DIR_NAME)
    appdata = encoding.GetEncodedValue(os.environ, 'APPDATA')
    if appdata:
        return os.path.join(appdata, _CLOUDSDK_GLOBAL_CONFIG_DIR_NAME)
    drive = encoding.GetEncodedValue(os.environ, 'SystemDrive', 'C:')
    return os.path.join(drive, os.path.sep, _CLOUDSDK_GLOBAL_CONFIG_DIR_NAME)