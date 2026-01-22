import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _EncodeTransform(self, match):
    char = match.group(0)
    if char == '\\':
        return '\\\\'
    if char == '"':
        return '\\"'
    return self._encode_transforms[ord(char)]