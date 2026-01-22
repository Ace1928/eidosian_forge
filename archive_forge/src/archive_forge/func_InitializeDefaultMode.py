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
def InitializeDefaultMode():
    """Records the default POSIX mode using os.umask."""
    global SYSTEM_POSIX_MODE
    if IS_WINDOWS:
        SYSTEM_POSIX_MODE = '666'
        return
    max_permissions = 511
    current_umask = os.umask(127)
    os.umask(current_umask)
    mode = max_permissions - current_umask
    SYSTEM_POSIX_MODE = oct(mode & 438)[-3:]