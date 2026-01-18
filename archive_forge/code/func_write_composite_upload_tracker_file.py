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
def write_composite_upload_tracker_file(tracker_file_path, random_prefix, encryption_key_sha256=None):
    """Updates or creates a tracker file for a composite upload.

  Args:
    tracker_file_path (str): The path to the tracker file.
    random_prefix (str): A prefix used to ensure temporary component names are
        unique across multiple running instances of the CLI.
    encryption_key_sha256 (str|None): The encryption key used for the
        upload.

  Returns:
    None, but writes data passed as arguments at tracker_file_path.
  """
    data = CompositeUploadTrackerData(encryption_key_sha256=encryption_key_sha256, random_prefix=random_prefix)
    _write_json_to_tracker_file(tracker_file_path, data._asdict())