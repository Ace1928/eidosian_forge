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
def _CrcMultiply(p, q):
    """Multiplies two polynomials together modulo CASTAGNOLI_POLY.

  Args:
    p: The first polynomial.
    q: The second polynomial.

  Returns:
    Result of the multiplication.
  """
    result = 0
    top_bit = 1 << DEGREE
    for _ in range(DEGREE):
        if p & 1:
            result ^= q
        q <<= 1
        if q & top_bit:
            q ^= CASTAGNOLI_POLY
        p >>= 1
    return result