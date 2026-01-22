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
def ValidateFilePermissionAccess(url_str, uid=NA_ID, gid=NA_ID, mode=NA_MODE):
    """Validates that the user has file access if uid, gid, and mode are applied.

  Args:
    url_str: The path to the object for which this is validating.
    uid: A POSIX user ID.
    gid: A POSIX group ID.
    mode: A 3-digit, number representing POSIX permissions, must be in base-8.

  Returns:
    A (bool, str) tuple, True if and only if it's safe to copy the file, and a
    string containing details for the error.
  """
    if IS_WINDOWS:
        return (True, '')
    uid_present = uid > NA_ID
    gid_present = int(gid) > NA_ID
    mode_present = mode > NA_MODE
    if not (uid_present or gid_present or mode_present):
        return (True, '')
    if os.geteuid() == 0:
        return (True, '')
    mode_valid = ValidatePOSIXMode(int(str(mode), 8))
    if mode_present:
        if not mode_valid:
            return (False, "Mode for %s won't allow read access." % url_str)
    else:
        mode = int(SYSTEM_POSIX_MODE)
    if uid_present:
        try:
            pwd.getpwuid(uid)
        except (KeyError, OverflowError):
            return (False, "UID for %s doesn't exist on current system. uid: %d" % (url_str, uid))
    if gid_present:
        try:
            grp.getgrgid(gid)
        except (KeyError, OverflowError):
            return (False, "GID for %s doesn't exist on current system. gid: %d" % (url_str, gid))
    uid_is_current_user = uid == os.getuid()
    mode = int(str(mode), 8)
    if not uid_present and gid_present and mode_present and (not bool(mode & U_R)):
        return (False, 'Insufficient access with uid/gid/mode for %s, gid: %d, mode: %s' % (url_str, gid, oct(mode)[-3:]))
    if uid_is_current_user:
        valid = bool(mode & U_R)
        return (valid, '' if valid else 'Insufficient access with uid/gid/mode for %s, uid: %d, mode: %s' % (url_str, uid, oct(mode)[-3:]))
    elif int(gid) in USER_GROUPS:
        valid = bool(mode & G_R)
        return (valid, '' if valid else 'Insufficient access with uid/gid/mode for %s, gid: %d, mode: %s' % (url_str, gid, oct(mode)[-3:]))
    elif mode & O_R:
        return (True, '')
    elif not uid_present and (not gid_present) and mode_valid:
        return (True, '')
    return (False, 'There was a problem validating %s.' % url_str)