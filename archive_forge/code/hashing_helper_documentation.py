from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import hashlib
import os
import six
from boto import config
import crcmod
from gslib.exception import CommandException
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
Catches up hashes, but does not return data and uses little memory.

    Before calling this function, digesters_current_mark should be updated
    to the current location of the original stream and the self._digesters
    should be current to that point (but no further).

    Args:
      bytes_to_read: Number of bytes to catch up from the original stream.
    