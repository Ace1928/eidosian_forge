import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _EncodeComment(self, comment):
    """Encodes a comment to be placed in the project file output, mimicking
    Xcode behavior.
    """
    return '/* ' + comment.replace('*/', '(*)/') + ' */'