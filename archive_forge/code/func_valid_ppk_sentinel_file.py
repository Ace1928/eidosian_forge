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
@property
def valid_ppk_sentinel_file(self):
    """Gets the path to the sentinel used to check for PPK encoding validity.

    The presence of this file is simply used to indicate whether or not we've
    correctly encoded the PPK used for ssh on Windows (re-encoding may be
    necessary in order to fix a bug in an older version of winkeygen.exe).

    Returns:
      str, The path to the sentinel file.
    """
    return os.path.join(self.global_config_dir, '.valid_ppk_sentinel')