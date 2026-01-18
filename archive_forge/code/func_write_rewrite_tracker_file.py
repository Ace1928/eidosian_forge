from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import hashlib
import json
import os
import re
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import scaled_integer
def write_rewrite_tracker_file(tracker_file_name, rewrite_parameters_hash, rewrite_token):
    """Writes rewrite operation information to a tracker file.

  Args:
    tracker_file_name (str): The path to the tracker file.
    rewrite_parameters_hash (str): MD5 hex digest of rewrite call parameters.
    rewrite_token (str): Returned by API, so rewrites can resume where they left
      off.
  """
    _write_tracker_file(tracker_file_name, '{}\n{}'.format(rewrite_parameters_hash, rewrite_token))