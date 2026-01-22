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
def Base64Sha256FromBase64EncryptionKey(csek_encryption_key):
    if six.PY3:
        if not isinstance(csek_encryption_key, bytes):
            csek_encryption_key = csek_encryption_key.encode('ascii')
    decoded_bytes = base64.b64decode(csek_encryption_key)
    key_sha256 = _CalculateSha256FromString(decoded_bytes)
    sha256_bytes = binascii.unhexlify(key_sha256)
    sha256_base64 = base64.b64encode(sha256_bytes)
    return sha256_base64.replace(b'\n', b'')