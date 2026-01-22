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
def CryptoKeyWrapperFromKey(crypto_key):
    """Returns a CryptoKeyWrapper for crypto_key, or None for no key."""
    return CryptoKeyWrapper(crypto_key) if crypto_key else None