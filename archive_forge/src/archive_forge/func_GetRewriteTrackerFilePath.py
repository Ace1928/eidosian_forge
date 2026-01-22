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
def GetRewriteTrackerFilePath(src_bucket_name, src_obj_name, dst_bucket_name, dst_obj_name, api_selector):
    """Gets the tracker file name described by the arguments.

  Args:
    src_bucket_name: Source bucket (string).
    src_obj_name: Source object (string).
    dst_bucket_name: Destination bucket (string).
    dst_obj_name: Destination object (string)
    api_selector: API to use for this operation.

  Returns:
    File path to tracker file.
  """
    res_tracker_file_name = re.sub('[/\\\\]', '_', 'rewrite__%s__%s__%s__%s__%s.token' % (src_bucket_name, src_obj_name, dst_bucket_name, dst_obj_name, api_selector))
    return _HashAndReturnPath(res_tracker_file_name, TrackerFileType.REWRITE)