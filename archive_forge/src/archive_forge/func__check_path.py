from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _check_path():
    """Checks for content aware metadata.

  If content aware metadata exists, return its absolute path;
  otherwise, returns None.

  Returns:
    str: Absolute path if exists. Otherwise, None.
  """
    metadata_path = os.path.expanduser(_DEFAULT_METADATA_PATH)
    if not os.path.exists(metadata_path):
        return None
    return metadata_path