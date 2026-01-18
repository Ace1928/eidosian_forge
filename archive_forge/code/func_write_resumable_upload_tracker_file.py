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
def write_resumable_upload_tracker_file(tracker_file_path, complete, encryption_key_sha256, serialization_data):
    """Updates or creates a tracker file for a resumable upload.

  Args:
    tracker_file_path (str): The path to the tracker file.
    complete (bool): True if the upload is complete.
    encryption_key_sha256 (Optional[str]): The encryption key used for the
        upload.
    serialization_data (dict): Data used by API libraries to resume uploads.

  Returns:
    None, but writes data passed as arguments at tracker_file_path.
  """
    data = ResumableUploadTrackerData(complete=complete, encryption_key_sha256=encryption_key_sha256, serialization_data=serialization_data)
    _write_json_to_tracker_file(tracker_file_path, data._asdict())