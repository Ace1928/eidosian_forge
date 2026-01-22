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
Initialize the CryptoKeyWrapper.

    Args:
      crypto_key: Base64-encoded string of a CSEK, or the name of a Cloud KMS
          CMEK.

    Raises:
      CommandException: The specified crypto key was neither a CMEK key name nor
          a valid base64-encoded string of a CSEK.
    