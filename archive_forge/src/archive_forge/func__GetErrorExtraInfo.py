from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def _GetErrorExtraInfo(error_extra_info):
    """Serializes the extra info into a json string for logging.

  Args:
    error_extra_info: {str: json-serializable}, A json serializable dict of
      extra info that we want to log with the error. This enables us to write
      queries that can understand the keys and values in this dict.

  Returns:
    str, The value to pass to Clearcut or None.
  """
    if error_extra_info:
        return json.dumps(error_extra_info, sort_keys=True)
    return None