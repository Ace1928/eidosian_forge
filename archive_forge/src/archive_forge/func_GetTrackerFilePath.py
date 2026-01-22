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
def GetTrackerFilePath(dst_url, tracker_file_type, api_selector, src_url=None, component_num=None):
    """Gets the tracker file name described by the arguments.

  Args:
    dst_url: Destination URL for tracker file.
    tracker_file_type: TrackerFileType for this operation.
    api_selector: API to use for this operation.
    src_url: Source URL for the source file name for parallel uploads.
    component_num: Component number if this is a download component, else None.

  Returns:
    File path to tracker file.
  """
    if tracker_file_type == TrackerFileType.UPLOAD:
        res_tracker_file_name = re.sub('[/\\\\]', '_', 'resumable_upload__%s__%s__%s.url' % (dst_url.bucket_name, dst_url.object_name, api_selector))
    elif tracker_file_type == TrackerFileType.DOWNLOAD:
        res_tracker_file_name = re.sub('[/\\\\]', '_', 'resumable_download__%s__%s.etag' % (os.path.realpath(dst_url.object_name), api_selector))
    elif tracker_file_type == TrackerFileType.DOWNLOAD_COMPONENT:
        res_tracker_file_name = re.sub('[/\\\\]', '_', 'resumable_download__%s__%s__%d.etag' % (os.path.realpath(dst_url.object_name), api_selector, component_num))
    elif tracker_file_type == TrackerFileType.PARALLEL_UPLOAD:
        res_tracker_file_name = re.sub('[/\\\\]', '_', 'parallel_upload__%s__%s__%s__%s.url' % (dst_url.bucket_name, dst_url.object_name, src_url, api_selector))
    elif tracker_file_type == TrackerFileType.SLICED_DOWNLOAD:
        res_tracker_file_name = re.sub('[/\\\\]', '_', 'sliced_download__%s__%s.etag' % (os.path.realpath(dst_url.object_name), api_selector))
    elif tracker_file_type == TrackerFileType.REWRITE:
        raise NotImplementedError()
    return _HashAndReturnPath(res_tracker_file_name, tracker_file_type)