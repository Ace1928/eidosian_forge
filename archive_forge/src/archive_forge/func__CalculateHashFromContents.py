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
def _CalculateHashFromContents(fp, hash_alg):
    """Calculates a base64 digest of the contents of a seekable stream.

  This function resets the file pointer to position 0.

  Args:
    fp: An already-open file object.
    hash_alg: Instance of hashing class initialized to start state.

  Returns:
    Hash of the stream in hex string format.
  """
    hash_dict = {'placeholder': hash_alg}
    fp.seek(0)
    CalculateHashesFromContents(fp, hash_dict)
    fp.seek(0)
    return hash_dict['placeholder'].hexdigest()