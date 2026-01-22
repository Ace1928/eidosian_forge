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
def _CalculateSha256FromString(input_string):
    sha256_hash = sha256()
    sha256_hash.update(input_string)
    return sha256_hash.hexdigest()