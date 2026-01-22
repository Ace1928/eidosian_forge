from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
from hashlib import sha256
import re
import sys
import six
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
def GetEncryptionKeyWrapper(boto_config):
    """Returns a CryptoKeyWrapper for the configured encryption key.

  Reads in the value of the "encryption_key" attribute in boto_config, and if
  present, verifies it is a valid base64-encoded string and returns a
  CryptoKeyWrapper for it.

  Args:
    boto_config: (boto.pyami.config.Config) The boto config in which to check
        for a matching encryption key.

  Returns:
    CryptoKeyWrapper for the specified encryption key, or None if no encryption
    key was specified in boto_config.
  """
    encryption_key = boto_config.get('GSUtil', 'encryption_key', None)
    return CryptoKeyWrapper(encryption_key) if encryption_key else None