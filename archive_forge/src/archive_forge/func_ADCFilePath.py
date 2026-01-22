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
def ADCFilePath():
    """Gets the ADC default file path.

  Returns:
    str, The path to the default ADC file.
  """
    from google.auth import _cloud_sdk
    return _cloud_sdk.get_application_default_credentials_path()