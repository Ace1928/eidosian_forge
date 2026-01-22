from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import binascii
import codecs
import os
import sys
import io
import re
import locale
import collections
import random
import six
import string
from six.moves import urllib
from six.moves import range
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.utils.constants import UTF8
from gslib.utils.constants import WINDOWS_1252
from gslib.utils.system_util import IS_CP1252
def DecodeLongAsString(long_to_convert):
    """Decodes an encoded python long into an ASCII string.

  This is used for modeling S3 version_id's as apitools generation.

  Args:
    long_to_convert: long to convert to ASCII string. If this is already a
                     string, it is simply returned.

  Returns:
    String decoded from the input long.
  """
    unhexed = binascii.unhexlify(hex(long_to_convert)[2:].rstrip('L'))
    return six.ensure_str(unhexed)