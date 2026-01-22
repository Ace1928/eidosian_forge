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
def Base64EncodeHash(digest_value):
    """Returns the base64-encoded version of the input hex digest value."""
    encoded_bytes = base64.b64encode(binascii.unhexlify(digest_value))
    return encoded_bytes.rstrip(b'\n').decode(UTF8)