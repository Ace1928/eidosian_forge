import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _EncodeString(self, value):
    """Encodes a string to be placed in the project file output, mimicking
    Xcode behavior.
    """
    if _unquoted.search(value) and (not _quoted.search(value)):
        return value
    return '"' + _escaped.sub(self._EncodeTransform, value) + '"'