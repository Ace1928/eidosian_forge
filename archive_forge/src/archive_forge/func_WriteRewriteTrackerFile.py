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
def WriteRewriteTrackerFile(tracker_file_name, rewrite_params_hash, rewrite_token):
    """Writes a rewrite tracker file.

  Args:
    tracker_file_name: Tracker file path string.
    rewrite_params_hash: MD5 hex digest of rewrite call parameters constructed
        by HashRewriteParameters.
    rewrite_token: Rewrite token string returned by the service.
  """
    _WriteTrackerFile(tracker_file_name, '%s\n%s\n' % (rewrite_params_hash, rewrite_token))