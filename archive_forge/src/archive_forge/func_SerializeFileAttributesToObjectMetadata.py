from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from calendar import timegm
import getpass
import logging
import os
import re
import time
import six
from gslib.exception import CommandException
from gslib.tz_utc import UTC
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import SECONDS_PER_DAY
def SerializeFileAttributesToObjectMetadata(posix_attrs, custom_metadata, preserve_posix=False):
    """Takes a POSIXAttributes object and serializes it into custom metadata.

  Args:
    posix_attrs: A POSIXAttributes object.
    custom_metadata: A custom metadata object to serialize values into.
    preserve_posix: Whether or not to preserve POSIX attributes other than
                    mtime.
  """
    if posix_attrs.mtime != NA_TIME:
        CreateCustomMetadata(entries={MTIME_ATTR: posix_attrs.mtime}, custom_metadata=custom_metadata)
    if preserve_posix:
        if posix_attrs.atime != NA_TIME:
            CreateCustomMetadata(entries={ATIME_ATTR: posix_attrs.atime}, custom_metadata=custom_metadata)
        if posix_attrs.uid != NA_ID:
            CreateCustomMetadata(entries={UID_ATTR: posix_attrs.uid}, custom_metadata=custom_metadata)
        if posix_attrs.gid != NA_ID:
            CreateCustomMetadata(entries={GID_ATTR: posix_attrs.gid}, custom_metadata=custom_metadata)
        if posix_attrs.mode.permissions != NA_MODE:
            CreateCustomMetadata(entries={MODE_ATTR: posix_attrs.mode.permissions}, custom_metadata=custom_metadata)