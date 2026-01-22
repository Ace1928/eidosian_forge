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
def RaiseUnwritableTrackerFileException(tracker_file_name, error_str):
    """Raises an exception when unable to write the tracker file."""
    raise CommandException(TRACKER_FILE_UNWRITABLE_EXCEPTION_TEXT % (tracker_file_name, error_str))