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
def GetUploadHashAlgs():
    """Returns a dict of hash algorithms for validating an uploaded object.

  This is for use only with single object uploads, not compose operations
  such as those used by parallel composite uploads (though it can be used to
  validate the individual components).

  Returns:
    dict of (algorithm_name: hash_algorithm)
  """
    check_hashes_config = config.get('GSUtil', 'check_hashes', CHECK_HASH_IF_FAST_ELSE_FAIL)
    if check_hashes_config == 'never':
        return {}
    return {'md5': GetMd5}