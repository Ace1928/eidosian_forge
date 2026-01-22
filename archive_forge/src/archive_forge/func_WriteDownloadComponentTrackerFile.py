from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import hashlib
import json
import os
import re
import sys
import six
from boto import config
from gslib.exception import CommandException
from gslib.utils.boto_util import GetGsutilStateDir
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.system_util import CreateDirIfNeeded
def WriteDownloadComponentTrackerFile(tracker_file_name, src_obj_metadata, current_file_pos):
    """Updates or creates a download component tracker file on disk.

  Args:
    tracker_file_name: The name of the tracker file.
    src_obj_metadata: Metadata for the source object. Must include etag.
    current_file_pos: The current position in the file.
  """
    component_data = {'etag': src_obj_metadata.etag, 'generation': src_obj_metadata.generation, 'download_start_byte': current_file_pos}
    _WriteTrackerFile(tracker_file_name, json.dumps(component_data))