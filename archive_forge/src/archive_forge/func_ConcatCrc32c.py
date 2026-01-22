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
def ConcatCrc32c(crc_a, crc_b, num_bytes_in_b):
    """Computes CRC32C for concat(A, B) given crc(A), crc(B) and len(B).

  An explanation of the algorithm can be found at
  crcutil.googlecode.com/files/crc-doc.1.0.pdf.

  Args:
    crc_a: A 32-bit integer representing crc(A) with least-significant
           coefficient first.
    crc_b: Same as crc_a.
    num_bytes_in_b: Length of B in bytes.

  Returns:
    CRC32C for concat(A, B)
  """
    if not num_bytes_in_b:
        return crc_a
    return _ExtendByZeros(crc_a, 8 * num_bytes_in_b) ^ crc_b